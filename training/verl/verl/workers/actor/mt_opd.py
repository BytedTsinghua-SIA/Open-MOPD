"""Multi-Teacher OPD (MT-OPD) utilities.

Domain-routing helpers for regular OPD (not delta-OPD): per sample, select the
RL teacher matching ``non_tensor_batch["domain"]``, then run the standard
``compute_distillation_reward`` path on the routed teacher log-probs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

__all__ = [
    "build_domain_weights",
    "select_routed_teacher_logprobs",
    "compute_domain_share_metrics",
    "compute_teacher_conflict_metrics",
    "compute_domain_loss_weights",
    "apply_teacher_conflict_policy",
]


def build_domain_weights(
    domains: Sequence[str] | np.ndarray | None,
    domain_order: list[str],
    device: torch.device | str,
    weighting: str = "domain_routing",
) -> torch.Tensor:
    """Build a per-sample teacher weight matrix of shape ``[B, N]``.

    Args:
        domains: String domain label per sample (length ``B``).  If ``None``
            falls back to uniform weighting regardless of ``weighting``.
        domain_order: Ordered list of domain names that matches the teacher
            index ordering (e.g. ``["math", "code", "if"]``).
        device: Target PyTorch device for the returned tensor.
        weighting: Strategy for assigning weights.

            * ``"domain_routing"`` — one-hot: the teacher whose domain matches
              gets weight 1, all others get 0.  Samples whose domain is not in
              ``domain_order`` fall back to uniform.
            * ``"uniform"`` — equal weight ``1/N`` for every teacher.

    Returns:
        Float32 tensor of shape ``[B, N]``.
    """
    n = len(domain_order)
    b = len(domains) if domains is not None else 1

    if domains is None or weighting == "uniform":
        return torch.full((b, n), 1.0 / n, dtype=torch.float32, device=device)

    weights = torch.zeros(b, n, dtype=torch.float32, device=device)
    domain_to_idx: dict[str, int] = {d: i for i, d in enumerate(domain_order)}
    for sample_idx, d in enumerate(domains):
        teacher_idx = domain_to_idx.get(str(d), -1)
        if teacher_idx >= 0:
            weights[sample_idx, teacher_idx] = 1.0
        else:
            weights[sample_idx] = 1.0 / n
    return weights


def select_routed_teacher_logprobs(
    teacher_logprobs: list[torch.Tensor],
    domain_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine per-teacher log-probs into a single routed tensor ``[B, T, K]``.

    For domain routing (one-hot rows) this picks one teacher per sample; for
    uniform weighting it averages across teachers.
    """
    n = len(teacher_logprobs)
    if domain_weights.shape[-1] != n:
        raise ValueError(
            f"domain_weights last dim ({domain_weights.shape[-1]}) must equal "
            f"number of teachers ({n})"
        )
    ref = teacher_logprobs[0]
    if ref.dim() != 3:
        raise ValueError(f"MT-OPD expects [B, T, K] tensors, got {tuple(ref.shape)}")

    stacked = torch.stack(teacher_logprobs, dim=0)  # [N, B, T, K]
    w = domain_weights.to(dtype=stacked.dtype, device=stacked.device)
    w = w.transpose(0, 1).reshape(n, -1, 1, 1)  # [N, B, 1, 1]
    return (w * stacked).sum(dim=0)


def compute_domain_share_metrics(
    domains: Sequence[str] | np.ndarray | None,
    response_mask: torch.Tensor,
    rm_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    """Per-domain prompt share, token share, length and reward scale.

    Without these a domain-weight sweep cannot be interpreted. Under
    ``loss_agg_mode=token-mean`` every token contributes equally to the loss, so a
    domain's gradient share tracks its **token** share, not its prompt share --
    and with response lengths differing by ~32x across domains the two diverge by
    two orders of magnitude. ``token_share`` is therefore the quantity a mixture
    ratio is actually setting.

    ``reward_abs_mean`` is reported per domain because gradient share is really
    ``token_share x reward_magnitude``: equalising tokens alone does not equalise
    the gradient if one domain's teacher-student gap is much larger.

    Args:
        domains: string domain label per sample, length ``B``.
        response_mask: ``[B, T]`` mask of real response tokens.
        rm_scores: optional ``[B, T]`` or ``[B, T, K]`` reward tensor.

    Returns:
        Flat metric dict, keys prefixed ``mt_opd/domain/``. Empty if no labels.
    """
    if domains is None or len(domains) == 0:
        return {}
    labels = [str(d) for d in domains]
    if response_mask.shape[0] != len(labels):
        raise ValueError(
            f"response_mask batch {response_mask.shape[0]} does not match "
            f"{len(labels)} domain labels"
        )

    mask = response_mask.detach()
    per_seq_tokens = mask.sum(dim=-1).float()
    total_tokens = per_seq_tokens.sum().clamp(min=1.0)
    batch_size = len(labels)

    reward_per_seq = None
    if rm_scores is not None:
        r = rm_scores.detach()
        if r.dim() == 3:
            r = r.sum(dim=-1)
        # abs, because the OPD reward is signed and a mean near zero would hide
        # a large magnitude that still drives the gradient.
        reward_per_seq = (r.abs() * mask).sum(dim=-1)

    out: dict[str, float] = {}
    for dom in sorted(set(labels)):
        idx = torch.tensor(
            [i for i, d in enumerate(labels) if d == dom],
            device=mask.device,
            dtype=torch.long,
        )
        tok = per_seq_tokens[idx].sum()
        p = f"mt_opd/domain/{dom}"
        out[f"{p}/prompt_share"] = float(idx.numel()) / batch_size
        out[f"{p}/token_share"] = (tok / total_tokens).item()
        out[f"{p}/prompt_count"] = float(idx.numel())
        out[f"{p}/mean_response_length"] = (tok / max(idx.numel(), 1)).item()
        if reward_per_seq is not None:
            out[f"{p}/reward_abs_mean"] = (
                reward_per_seq[idx].sum() / tok.clamp(min=1.0)
            ).item()
    return out


def compute_domain_loss_weights(
    domains: Sequence[str] | np.ndarray | None,
    response_mask: torch.Tensor,
    target_shares: dict[str, float] | None = None,
    rm_scores: torch.Tensor | None = None,
    normalize_reward_scale: bool | float = False,
    reward_scale_stat: str = "mean",
    reward_scale_direction: str = "divide",
    reward_anchor: dict[str, float] | None = None,
) -> torch.Tensor | None:
    """Per-sequence loss weights that hit a target *gradient* share per domain.

    Motivation, measured: student response lengths are math 11570 / code 9032 /
    IF 364 tokens. Under ``loss_agg_mode=token-mean`` every token contributes
    equally, so a domain's gradient share is its token share -- which made IF hold
    20% of prompts but 0.9% of the gradient, and made the 2:2:1 -> 3:3:1 prompt
    sweep move token share by only 0.1-0.3pp.

    Reaching equal share by sampling would need ~32x more IF prompts than math,
    which does not fit in a batch. Scaling the loss reaches any target share
    without touching sampling: weight domain ``d`` by ``target_d / observed_d``.

    The observed share is taken from **this batch**, so the correction is
    self-calibrating and does not depend on the length estimates above staying
    true as the student's length drifts during training.

    Args:
        domains: string label per sample, length ``B``.
        response_mask: ``[B, T]``.
        target_shares: domain -> desired gradient share. Missing domains default
            to an equal split across those present. ``None`` returns ``None``,
            meaning "leave the loss alone".
        rm_scores: reward tensor, required when ``normalize_reward_scale``.
        normalize_reward_scale: **M2**. Gradient share is really
            ``token_share x reward_magnitude``, so equalising tokens alone leaves a
            domain with a larger teacher-student gap still dominating. When set,
            each domain's weight is additionally divided by its own mean absolute
            reward per token, making the *product* match the target instead of just
            the token part. Composes with M1 rather than replacing it: with token
            share spanning two orders of magnitude and reward scale typically well
            under one, M1 is the larger correction, and that ordering is a
            prediction the two runs test separately.

    Returns:
        ``[B]`` float tensor of per-sequence weights, mean-normalised to 1 so the
        overall loss scale (and thus the effective learning rate) is unchanged and
        only the *relative* domain shares move. ``None`` if not applicable.
    """
    if target_shares is None or domains is None or len(domains) == 0:
        return None
    if normalize_reward_scale and rm_scores is None:
        raise ValueError("normalize_reward_scale=True requires rm_scores")
    labels = [str(d) for d in domains]
    if response_mask.shape[0] != len(labels):
        raise ValueError(
            f"response_mask batch {response_mask.shape[0]} does not match "
            f"{len(labels)} domain labels"
        )

    mask = response_mask.detach()
    per_seq = mask.sum(dim=-1).float()
    total = per_seq.sum().clamp(min=1.0)
    present = sorted(set(labels))

    # Normalise the requested targets over the domains actually in this batch;
    # a domain absent from the batch cannot be given gradient share here.
    raw = {d: float(target_shares.get(d, 0.0)) for d in present}
    if sum(raw.values()) <= 0:
        raw = {d: 1.0 / len(present) for d in present}
    tsum = sum(raw.values())
    target = {d: raw[d] / tsum for d in present}

    # M2 strength: bool True == 1.0 == full division by reward magnitude. A float in
    # (0, 1) applies a *partial* correction, mag**alpha. Measured at alpha=1 the
    # correction overshoots -- math weight went 0.697 -> 1.326 (~1.9x) while code went
    # 0.654 -> 0.333 (~0.5x), and math/code both lost 2.2pp -- so the useful setting
    # may be interior rather than at either endpoint. alpha=0 recovers M1 exactly.
    alpha = 1.0 if normalize_reward_scale is True else float(normalize_reward_scale or 0.0)
    if not (0.0 <= alpha <= 2.0):
        raise ValueError(f"normalize_reward_scale alpha must be in [0, 2], got {alpha}")
    if reward_scale_stat not in ("mean", "q75", "q90", "top10"):
        raise ValueError(f"reward_scale_stat must be mean|q75|q90|top10, got {reward_scale_stat}")
    if reward_scale_direction not in ("divide", "multiply"):
        raise ValueError(
            f"reward_scale_direction must be divide|multiply, got {reward_scale_direction}"
        )
    if reward_anchor is not None:
        missing = [d for d in present if d not in reward_anchor]
        if missing:
            raise ValueError(f"reward_anchor missing domains present in batch: {missing}")

    # M2 length-dilution fix (measured 2026-08-04): per-token |reward| correlates with
    # response length at r=-0.87 (math) / -0.80 (code), because long generations are
    # mostly low-signal continuation tokens. A *mean* over tokens therefore reads math's
    # scale ~3.4x below code/IF even where it matters. A high quantile reads the
    # high-signal positions that actually carry the domain and is filler-immune.
    reward_per_token = None
    reward_values = None
    if alpha > 0.0:
        r = rm_scores.detach()
        if r.dim() == 3:
            r = r.sum(dim=-1)
        if reward_scale_stat == "mean":
            reward_per_token = (r.abs() * mask).sum(dim=-1)
        else:
            reward_values = r.abs() * mask  # [B, T], zeros outside the response

    weights = torch.ones(len(labels), dtype=torch.float32, device=mask.device)
    # Pass 1: per-domain token share + reward magnitude.
    info: dict[str, tuple[list[int], float, float | None]] = {}
    for dom in present:
        idx = [i for i, d in enumerate(labels) if d == dom]
        observed = (per_seq[idx].sum() / total).item()
        if observed <= 0:
            continue
        mag = None
        if reward_per_token is not None:
            tok = per_seq[idx].sum().clamp(min=1.0)
            mag = (reward_per_token[idx].sum() / tok).item()
        elif reward_values is not None:
            vals = reward_values[idx][mask[idx].bool()]
            vals = vals[vals > 0]
            if vals.numel() > 0:
                if reward_scale_stat == "top10":
                    k = max(1, vals.numel() // 10)
                    mag = vals.float().topk(k).values.mean().item()
                else:
                    q = 0.75 if reward_scale_stat == "q75" else 0.90
                    mag = torch.quantile(vals.float(), q).item()
        info[dom] = (idx, target[dom] / observed, mag)

    # M2 direction. "divide" (the original, measured harmful 2026-08-04/06): divide
    # by current magnitude -- at rest this up-weights the quiet domain, but over
    # training the easy domain's gap collapses ~35x while math/code collapse ~2x,
    # so 1/mag**alpha snowballs onto the *easiest* domain (measured: IF loss
    # weight 27 -> 91, grad token share 57% vs target 33%). "multiply" multiplies
    # by the magnitude ratio instead, i.e. gradient budget follows the *remaining*
    # teacher gap: a domain whose gap has collapsed loses share automatically.
    # - unanchored: reference is the running M1-weighted mean magnitude.
    # - anchored ("anchored" via reward_anchor): reference is this run's own
    #   step-1 magnitude per domain, so step 1 is exactly M1 and only the *drift*
    #   moves shares; guards anchored experiments from M2's static re-shuffle.
    mags = {d: m for d, (idx, sc, m) in info.items() if m is not None}
    if alpha > 0.0 and mags:
        if reward_scale_direction == "multiply":
            if reward_anchor is not None:
                ref = reward_anchor
            else:
                base = {d: sc for d, (_, sc, m) in info.items() if m is not None}
                bsum = sum(base.values())
                ref = {
                    d: max(sum(base[dd] * mags[dd] for dd in base) / max(bsum, 1e-12), 1e-12)
                    for d in mags
                }
        for dom in list(mags):
            m = mags[dom]
            idx, sc, mag0 = info[dom]
            if reward_scale_direction == "divide":
                # Original semantics preserved: skip the correction on a vanishing
                # reward (guard against an unbounded weight).
                if m <= 1e-8:
                    continue
                sc = sc / (m ** alpha)
            else:
                fac = (max(m, 1e-12) / max(ref[dom], 1e-12)) ** alpha
                fac = min(max(fac, 0.05), 20.0)
                sc = sc * fac
            info[dom] = (idx, sc, mag0)

    for dom, (idx, scale, _mag) in info.items():
        for i in idx:
            weights[i] = scale

    # Token-weighted mean of 1: preserves total loss magnitude, shifts only shares.
    denom = (weights * per_seq).sum().clamp(min=1e-6)
    weights = weights * (total / denom)
    return weights


def refresh_opd_advantage(
    student_top_k_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    reward_weight_mode: str = "student_p",
) -> torch.Tensor:
    """**M4 (AsyncOPD-style).** Rebuild the OPD advantage from the *current* student.

    Why this exists. Off-policy OPD has two distinct staleness terms, and they are not
    equally fixable:

      1. *state-distribution* staleness -- the tokens were sampled by an older policy.
         Unfixable without generating again.
      2. *reward* staleness -- the reward is ``-(S_logp - T_on_S) * w(S_logp)``, which
         depends on the student's **own current** log-prob. After an inner update
         ``S_logp`` has moved, so the stored reward describes a student that no longer
         exists. Importance sampling cannot repair this: IS corrects a wrong *sampling
         distribution*, not a wrong *reward*.

    Term 2 is free to fix. ``T_on_S`` comes from a frozen teacher on already-sampled ids,
    so it stays valid forever and can be cached; the only stale factor is ``S_logp``, and
    ``update_policy`` already recomputes exactly that (on the same ``student_top_k_ids``)
    for the gradient. So the refresh costs **no extra forward pass** -- it reuses a tensor
    the backward pass needed anyway. This is the cheap half of AsyncOPD's "recompute the
    advantage at learner time", which reports 1.6-3.8x throughput at matched accuracy.

    Only valid for ``top_k_strategy='only_stu'``, where the reward is a closed form in
    ``(S_logp, T_on_S)`` with no teacher-side ids to score.

    Args:
        student_top_k_log_probs: ``[B, T, K]`` student log-probs on its own sampled
            top-k ids, from the current weights. Pass a **detached** tensor: this is a
            reward, and letting gradient flow through it would change the objective.
        teacher_on_student_log_probs: ``[B, T, K]`` cached frozen-teacher log-probs on
            those same ids.
        response_mask: ``[B, T]``.
        reward_weight_mode: must match the rollout-time setting.

    Returns:
        ``[B, T, K]`` refreshed advantage, identical in construction to what
        ``compute_distillation_reward`` + ``token_reward_direct`` produce at step 0.
    """
    if student_top_k_log_probs.shape != teacher_on_student_log_probs.shape:
        raise ValueError(
            f"student {tuple(student_top_k_log_probs.shape)} and teacher "
            f"{tuple(teacher_on_student_log_probs.shape)} top-k log-probs must match"
        )
    s = student_top_k_log_probs.detach()
    t = teacher_on_student_log_probs.to(s.dtype)

    if reward_weight_mode == "student_p":
        logits = s
    elif reward_weight_mode == "teacher_p":
        logits = t
    elif reward_weight_mode == "none":
        logits = torch.zeros_like(s)
    else:
        raise ValueError(f"Unknown reward_weight_mode: {reward_weight_mode}")

    # only_stu has an all-ones valid mask, so the softmax needs no -inf masking.
    weights = torch.exp(logits - torch.logsumexp(logits, dim=-1, keepdim=True))
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    rm_scores = -(s - t) * weights
    return rm_scores * response_mask.unsqueeze(-1).to(rm_scores.dtype)


def apply_teacher_conflict_policy(
    routed_logprobs: torch.Tensor,
    teacher_logprobs: list[torch.Tensor],
    response_mask: torch.Tensor,
    policy: str = "none",
    conflict_nats: float = 1.0,
    conflict_quantile: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    """**M3**: use cross-teacher agreement, which routing currently throws away.

    The MT path evaluates every teacher on every student token and keeps only the
    routed one, so disagreement is already paid for. This is the only signal in the
    recipe that a single mixed-reward run (MixRL) structurally cannot have: MixRL
    sees one scalar per domain, never three opinions on the same token.

    Policies:
        ``none``
            Return the routed target unchanged. Default, so MT is untouched.
        ``mask``
            Zero the per-token loss weight where teachers disagree by more than
            ``conflict_nats``. On a contested token the routed teacher's target is
            not a consensus, so the gradient there is at least partly arbitrary.
        ``consensus``
            Replace the routed target with the across-teacher **mean** on tokens
            where teachers agree, keeping the routed target where they do not. On
            agreeing tokens the mean is a lower-variance estimate of the same thing,
            for free; on contested tokens averaging would blur a real distinction,
            which is exactly where the routed teacher should win.

    Returns:
        ``(target_logprobs, token_weight_or_None, metrics)``. ``token_weight`` is
        ``[B, T]`` and multiplies the per-token loss; ``None`` means unweighted.
    """
    if policy not in {"none", "mask", "consensus"}:
        raise ValueError(f"unknown teacher conflict policy: {policy!r}")
    if policy == "none" or teacher_logprobs is None or len(teacher_logprobs) < 2:
        return routed_logprobs, None, {}

    stacked = torch.stack([t.detach() for t in teacher_logprobs], dim=0)
    if stacked.shape[1:] != routed_logprobs.shape:
        raise ValueError(
            f"teacher tensors {tuple(stacked.shape[1:])} do not match routed "
            f"{tuple(routed_logprobs.shape)}"
        )
    top1 = stacked[..., 0].float()
    spread = top1.max(dim=0).values - top1.min(dim=0).values   # [B, T]

    mask = response_mask.detach().float()
    denom = mask.sum().clamp(min=1.0)

    # A quantile threshold is self-calibrating: measured mean spread is 0.131 nats
    # against a max of 29.2, so any fixed nats value either catches almost nothing
    # (1.0 nats -> 0.70% of tokens) or has to be guessed against a distribution that
    # drifts as the student moves. The quantile pins the *fraction* treated as
    # contested instead, which is the quantity the policy actually trades off.
    thresh = conflict_nats
    if conflict_quantile is not None:
        if not 0.0 < conflict_quantile < 1.0:
            raise ValueError(
                f"conflict_quantile must be in (0, 1), got {conflict_quantile}"
            )
        real = spread[mask.bool()]
        if real.numel():
            thresh = torch.quantile(real, conflict_quantile).item()

    agree = (spread <= thresh)
    metrics = {
        "mt_opd/conflict/policy_agree_frac": ((agree.float() * mask).sum() / denom).item(),
        "mt_opd/conflict/policy_threshold_nats": float(thresh),
    }

    if policy == "mask":
        weight = agree.to(routed_logprobs.dtype)
        # Renormalise so dropping tokens does not quietly shrink the total loss
        # (and with it the effective learning rate).
        kept = (weight * mask).sum().clamp(min=1.0)
        weight = weight * (denom / kept)
        metrics["mt_opd/conflict/mask_rescale"] = (denom / kept).item()
        return routed_logprobs, weight, metrics

    # consensus
    mean_lp = stacked.mean(dim=0)
    target = torch.where(
        agree.unsqueeze(-1).expand_as(routed_logprobs), mean_lp, routed_logprobs
    )
    metrics["mt_opd/conflict/consensus_frac"] = metrics["mt_opd/conflict/policy_agree_frac"]
    return target, None, metrics


def compute_teacher_conflict_metrics(
    teacher_logprobs: list[torch.Tensor],
    response_mask: torch.Tensor,
    domains: Sequence[str] | np.ndarray | None = None,
    conflict_nats: float = 1.0,
) -> dict[str, float]:
    """Cross-teacher disagreement on the student's own tokens.

    This costs nothing: the MT path already evaluates every teacher on every
    student token and then discards all but the routed one. The spread across
    teachers is the multi-teacher analogue of MixRL's opinion conflict, and it is
    information a single mixed-reward run structurally cannot have.

    Spread is measured in nats as ``max - min`` over teachers, per token, on the
    student's top-1 candidate (index 0 of the K axis) so the number refers to the
    token the student actually favours rather than an average over candidates.

    Args:
        teacher_logprobs: ``N`` tensors of ``[B, T, K]``, one per teacher.
        response_mask: ``[B, T]``.
        domains: optional labels, to report conflict per domain.
        conflict_nats: absolute threshold above which a token counts as contested.

    Returns:
        Flat metric dict prefixed ``mt_opd/conflict/``. Empty if fewer than two
        teachers, since disagreement is undefined for one.
    """
    # NOTE: measured on the instrumented MT run, mean spread is 0.131 nats and only
    # 0.70% of tokens exceed 1.0 nats, while spread_max reaches 29.2 -- conflict is
    # real but concentrated in a rare tail. Absolute thresholds are therefore hard to
    # pick blind, which is why the policy below also accepts a quantile.
    if teacher_logprobs is None or len(teacher_logprobs) < 2:
        return {}

    stacked = torch.stack([t.detach() for t in teacher_logprobs], dim=0)
    if stacked.dim() != 4:
        raise ValueError(
            f"expected [N, B, T, K] after stacking, got {tuple(stacked.shape)}"
        )
    top1 = stacked[..., 0].float()          # [N, B, T]
    spread = top1.max(dim=0).values - top1.min(dim=0).values  # [B, T]

    mask = response_mask.detach().float()
    denom = mask.sum().clamp(min=1.0)
    contested = (spread > conflict_nats).float()

    out = {
        "mt_opd/conflict/spread_mean": ((spread * mask).sum() / denom).item(),
        "mt_opd/conflict/contested_frac": ((contested * mask).sum() / denom).item(),
        "mt_opd/conflict/spread_max": (spread * mask).max().item(),
    }

    if domains is not None and len(domains) == mask.shape[0]:
        labels = [str(d) for d in domains]
        for dom in sorted(set(labels)):
            idx = torch.tensor(
                [i for i, d in enumerate(labels) if d == dom],
                device=mask.device,
                dtype=torch.long,
            )
            m = mask[idx]
            d_denom = m.sum().clamp(min=1.0)
            out[f"mt_opd/conflict/{dom}/spread_mean"] = (
                (spread[idx] * m).sum() / d_denom
            ).item()
            out[f"mt_opd/conflict/{dom}/contested_frac"] = (
                (contested[idx] * m).sum() / d_denom
            ).item()
    return out

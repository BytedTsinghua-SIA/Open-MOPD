"""M1: rebalance MT-OPD gradient share across domains.

Measured premise. Student response lengths are math 11570 / code 9032 / IF 364, and
under ``loss_agg_mode=token-mean`` every token contributes equally to the loss, so a
domain's gradient share is its *token* share:

    weights 2:2:1  -> prompt 40/40/20 %  but token 55.7/43.5/0.9 %
    weights 3:3:1  -> prompt 42.9/42.9/14.3 %  but token 55.8/43.6/0.6 %

That is why the prompt-weight sweep was null (token share moved 0.1-0.3pp) and why IF
is MT's weakest domain against RouteOPD (-4.71): it receives 0.9% of the gradient.
Equalising by sampling would need ~32x more IF prompts than math, which does not fit in
a batch, so the loss is scaled instead.
"""
import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import agg_loss
from verl.workers.actor.mt_opd import compute_domain_loss_weights


def _shares(domains, mask, weights=None):
    """Token share per domain, optionally after applying loss weights."""
    per_seq = mask.sum(dim=-1).float()
    if weights is not None:
        per_seq = per_seq * weights
    total = per_seq.sum()
    out = {}
    for d in sorted(set(domains)):
        idx = [i for i, x in enumerate(domains) if x == d]
        out[d] = (per_seq[idx].sum() / total).item()
    return out


def _mt_batch():
    """A batch with the measured length ratios: math/code long, IF very short."""
    domains = ["math", "math", "code", "code", "if"]
    lens = [11570, 11570, 9032, 9032, 364]
    T = max(lens)
    mask = torch.zeros(len(domains), T)
    for i, n in enumerate(lens):
        mask[i, :n] = 1.0
    return domains, mask


def test_the_problem_reproduces() -> None:
    """IF holds 20% of prompts and under 1% of tokens."""
    domains, mask = _mt_batch()
    s = _shares(domains, mask)
    assert s["if"] < 0.01, f"IF token share should be ~0.9%, got {s['if']:.4f}"
    assert 0.19 < 1 / len(domains) * len([d for d in domains if d == "if"]) <= 0.21


def test_equal_target_equalises_token_weighted_share() -> None:
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(
        domains, mask, {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3}
    )
    s = _shares(domains, mask, w)
    for d in ("math", "code", "if"):
        assert s[d] == pytest.approx(1 / 3, abs=1e-4), f"{d} share {s[d]:.4f}"


def test_if_weight_is_large_because_its_share_is_tiny() -> None:
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(
        domains, mask, {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3}
    )
    w_if = w[domains.index("if")].item()
    w_math = w[domains.index("math")].item()
    assert w_if / w_math > 20, (
        "the correction must be order-of-magnitude, matching the 0.9% -> 33% move; "
        f"got if/math = {w_if / w_math:.1f}"
    )


@pytest.mark.parametrize("target", [
    {"math": 0.5, "code": 0.3, "if": 0.2},
    {"math": 0.2, "code": 0.2, "if": 0.6},
    {"math": 0.8, "code": 0.1, "if": 0.1},
])
def test_arbitrary_targets_are_hit(target) -> None:
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(domains, mask, target)
    s = _shares(domains, mask, w)
    for d, want in target.items():
        assert s[d] == pytest.approx(want, abs=1e-4)


def test_unnormalised_targets_are_normalised() -> None:
    """Ratios, not fractions, must work -- 2:2:1 is how these are written."""
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(domains, mask, {"math": 2, "code": 2, "if": 1})
    s = _shares(domains, mask, w)
    assert s["math"] == pytest.approx(0.4, abs=1e-4)
    assert s["code"] == pytest.approx(0.4, abs=1e-4)
    assert s["if"] == pytest.approx(0.2, abs=1e-4)


def test_weights_preserve_total_loss_scale() -> None:
    """Token-weighted mean 1, so the effective learning rate does not move."""
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(
        domains, mask, {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3}
    )
    per_seq = mask.sum(dim=-1).float()
    assert (w * per_seq).sum().item() == pytest.approx(per_seq.sum().item(), rel=1e-5)


def test_none_target_leaves_loss_untouched() -> None:
    """The default path must be bit-identical to before this feature existed."""
    domains, mask = _mt_batch()
    assert compute_domain_loss_weights(domains, mask, None) is None
    assert compute_domain_loss_weights(None, mask, {"math": 1.0}) is None


def test_absent_domain_does_not_steal_share() -> None:
    """A batch may not contain every domain; targets renormalise over what is present."""
    domains = ["math", "if"]
    mask = torch.zeros(2, 100)
    mask[0, :90] = 1.0
    mask[1, :10] = 1.0
    w = compute_domain_loss_weights(
        domains, mask, {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3}
    )
    s = _shares(domains, mask, w)
    assert s["math"] == pytest.approx(0.5, abs=1e-4)
    assert s["if"] == pytest.approx(0.5, abs=1e-4)


def test_advantage_scaling_matches_mask_scaling() -> None:
    """The equivalence the implementation relies on.

    M1 scales the advantage rather than response_mask, because response_mask is also
    the averaging mask for ppo_kl and pg_clipfrac and scaling it would make those
    report almost only the up-weighted domain. That choice is only safe if the two are
    numerically equivalent, which holds because the weights have token-weighted mean 1
    and so leave token-mean's denominator unchanged.
    """
    domains, mask = _mt_batch()
    mask = mask[:, :200]                      # keep the tensor small
    w = compute_domain_loss_weights(domains, mask, {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3})

    g = torch.Generator().manual_seed(0)
    adv = torch.randn(mask.shape, generator=g)

    via_adv = agg_loss(loss_mat=-(adv * w.unsqueeze(-1)), loss_mask=mask,
                       loss_agg_mode="token-mean")
    via_mask = agg_loss(loss_mat=-adv, loss_mask=mask * w.unsqueeze(-1),
                        loss_agg_mode="token-mean")

    assert via_adv.item() == pytest.approx(via_mask.item(), rel=1e-5), (
        f"advantage scaling {via_adv.item():.6f} vs mask scaling {via_mask.item():.6f}"
    )


def test_accepts_numpy_labels() -> None:
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(
        np.array(domains, dtype=object), mask,
        {"math": 1 / 3, "code": 1 / 3, "if": 1 / 3},
    )
    assert w is not None and w.shape == (len(domains),)


def test_mismatched_batch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_domain_loss_weights(["a", "b"], torch.ones(3, 4), {"a": 1.0})


def test_zero_targets_fall_back_to_equal() -> None:
    """All-zero targets would divide by zero; fall back rather than emit NaN."""
    domains, mask = _mt_batch()
    w = compute_domain_loss_weights(domains, mask, {"math": 0.0, "code": 0.0, "if": 0.0})
    assert torch.isfinite(w).all()
    s = _shares(domains, mask, w)
    for d in ("math", "code", "if"):
        assert s[d] == pytest.approx(1 / 3, abs=1e-4)

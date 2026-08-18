"""M2 (reward-scale normalisation) and M3 (teacher-conflict policy).

M2 exists because gradient share is ``token_share x reward_magnitude``, so M1's token
correction alone still leaves a domain with a larger teacher-student gap dominating.
The ordering claim -- M1 is the larger correction, M2 only matters after it -- is a
prediction the two runs test separately, and these tests pin the mechanics that make
the claim meaningful.

M3 uses cross-teacher agreement. The MT path already evaluates every teacher on every
student token and keeps only the routed one, so this is free, and it is the only signal
in the recipe that MixRL structurally cannot have: MixRL sees one scalar reward per
domain, never three opinions on one token.
"""
import pytest
import torch

from verl.workers.actor.mt_opd import (
    apply_teacher_conflict_policy,
    compute_domain_loss_weights,
)


def _two_domain_batch():
    domains = ["math", "if"]
    mask = torch.zeros(2, 100)
    mask[0, :90] = 1.0    # math: long
    mask[1, :10] = 1.0    # IF: short
    return domains, mask


# --- M2 ---------------------------------------------------------------------

def test_reward_scale_requires_rm_scores() -> None:
    domains, mask = _two_domain_batch()
    with pytest.raises(ValueError, match="requires rm_scores"):
        compute_domain_loss_weights(
            domains, mask, {"math": 0.5, "if": 0.5}, normalize_reward_scale=True
        )


def test_reward_scale_downweights_the_loud_domain() -> None:
    """A domain whose reward is already large needs less weight for the same share."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 10.0     # math: large reward magnitude
    rm[1, :10] = 0.1      # IF: small

    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    w_m2 = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
    )
    r_m1 = (w_m1[1] / w_m1[0]).item()
    r_m2 = (w_m2[1] / w_m2[0]).item()
    assert r_m2 > r_m1, (
        "M2 must push further toward the quiet domain than M1 alone: "
        f"if/math weight ratio {r_m1:.2f} -> {r_m2:.2f}"
    )


def test_reward_scale_is_inert_when_magnitudes_match() -> None:
    """M2 must be a no-op when there is no reward-scale imbalance to correct."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 2.0
    rm[1, :10] = 2.0
    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    w_m2 = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
    )
    assert torch.allclose(w_m1, w_m2, rtol=1e-5), (
        "equal reward magnitudes must leave M1's weights untouched, or M2 is not "
        "separable from M1"
    )


def test_reward_scale_survives_a_vanishing_reward() -> None:
    """A near-zero reward must not produce an unbounded weight."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 1.0
    rm[1, :10] = 0.0      # IF reward collapsed
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
    )
    assert torch.isfinite(w).all() and (w.abs() < 1e6).all()


def test_reward_scale_preserves_total_loss_magnitude() -> None:
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 5.0
    rm[1, :10] = 0.5
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
    )
    per_seq = mask.sum(dim=-1)
    assert (w * per_seq).sum().item() == pytest.approx(per_seq.sum().item(), rel=1e-5)


def test_multiply_gives_the_loud_domain_moar_share() -> None:
    """M2v2 multiplies by the magnitude ratio: the loud domain gets MORE, the
    quiet one less -- the exact opposite of divide, on purpose (2026-08-06:
    divide-measured-harmful because it snowballs onto the fastest-learning
    domain as its gap collapses)."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 10.0     # math: large remaining gap
    rm[1, :10] = 0.1      # IF: nearly learned
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
        reward_scale_direction="multiply",
    )
    ratio = (w[0] / w[1]).item()
    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    ratio_m1 = (w_m1[0] / w_m1[1]).item()
    assert ratio > ratio_m1, (
        f"multiply must tilt toward the loud domain: math/if ratio {ratio:.3f} "
        f"vs M1 {ratio_m1:.4f}"
    )


def test_multiply_alpha_zero_is_exactly_m1() -> None:
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 10.0
    rm[1, :10] = 0.1
    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    for direction in ("divide", "multiply"):
        w = compute_domain_loss_weights(
            domains, mask, {"math": 0.5, "if": 0.5},
            rm_scores=rm, normalize_reward_scale=0.0,
            reward_scale_direction=direction,
        )
        assert torch.allclose(w, w_m1), f"{direction} with alpha=0 must equal M1"


def test_anchored_multiply_is_exactly_m1_at_the_anchor() -> None:
    """With the anchor equal to the current magnitudes, M2v2-anchored must be
    bit-identical to M1: step 1 of an anchored run is M1 by construction."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 2.0
    rm[1, :10] = 4.0
    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    anchor = {"math": 2.0, "if": 4.0}
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
        reward_scale_direction="multiply", reward_anchor=anchor,
    )
    assert torch.allclose(w, w_m1, rtol=1e-5), (
        "anchored at the current magnitudes must leave M1 untouched"
    )


def test_anchored_multiply_damps_the_domain_whose_gap_collapsed() -> None:
    """Anchored semantics: a domain whose gap fell 10x since the anchor loses
    weight by 10^alpha relative to a domain whose gap held."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 2.0   # math: gap held at anchor
    rm[1, :10] = 0.2   # IF: gap fell 10x since the anchor
    anchor = {"math": 2.0, "if": 2.0}
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=True,
        reward_scale_direction="multiply", reward_anchor=anchor,
    )
    w_m1 = compute_domain_loss_weights(domains, mask, {"math": 0.5, "if": 0.5})
    # IF's weight share must drop relative to M1.
    assert w[1].item() < w_m1[1].item(), (
        f"a 10x-collapsed domain must lose weight: if weight {w[1]:.1f} vs M1 {w_m1[1]:.1f}"
    )
    # Mean-normalization preserved.
    per_seq = mask.sum(dim=-1)
    assert (w * per_seq).sum().item() == pytest.approx(per_seq.sum().item(), rel=1e-5)


def test_multiply_factor_is_clamped() -> None:
    """A 1e6x magnitude gap must not produce an unbounded weight ratio."""
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 1e-6
    rm[1, :10] = 1.0
    w = compute_domain_loss_weights(
        domains, mask, {"math": 0.5, "if": 0.5},
        rm_scores=rm, normalize_reward_scale=1.0,
        reward_scale_direction="multiply",
    )
    assert torch.isfinite(w).all()
    per_seq = mask.sum(dim=-1)
    assert (w * per_seq).sum().item() == pytest.approx(per_seq.sum().item(), rel=1e-5)


def test_bad_direction_fails_loudly() -> None:
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    with pytest.raises(ValueError, match="divide"):
        compute_domain_loss_weights(
            domains, mask, {"math": 0.5, "if": 0.5},
            rm_scores=rm, normalize_reward_scale=True,
            reward_scale_direction="sideways",
        )


def test_anchor_must_cover_present_domains() -> None:
    domains, mask = _two_domain_batch()
    rm = torch.zeros(2, 100)
    rm[0, :90] = 1.0
    rm[1, :10] = 1.0
    with pytest.raises(ValueError, match="missing domains"):
        compute_domain_loss_weights(
            domains, mask, {"math": 0.5, "if": 0.5},
            rm_scores=rm, normalize_reward_scale=True,
            reward_scale_direction="multiply", reward_anchor={"math": 1.0},
        )


# --- M3 ---------------------------------------------------------------------

def _teachers(spread_nats, b=1, t=4, k=3):
    a = torch.zeros(b, t, k)
    c = torch.zeros(b, t, k)
    c[..., 0] = -spread_nats
    return [a, c]


def test_policy_none_is_bit_identical() -> None:
    """The default must leave MT exactly as it was."""
    routed = torch.randn(1, 4, 3)
    tl = _teachers(5.0)
    out, weight, m = apply_teacher_conflict_policy(
        routed, tl, torch.ones(1, 4), policy="none"
    )
    assert out is routed and weight is None and m == {}


def test_unknown_policy_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown teacher conflict policy"):
        apply_teacher_conflict_policy(
            torch.zeros(1, 2, 2), _teachers(1.0, t=2, k=2), torch.ones(1, 2),
            policy="averge",
        )


def test_single_teacher_is_a_noop() -> None:
    """Disagreement is undefined for one teacher."""
    routed = torch.randn(1, 3, 2)
    out, weight, m = apply_teacher_conflict_policy(
        routed, [torch.randn(1, 3, 2)], torch.ones(1, 3), policy="consensus"
    )
    assert out is routed and weight is None and m == {}


def test_mask_zeroes_contested_tokens() -> None:
    routed = torch.randn(1, 4, 3)
    tl = _teachers(0.0)
    tl[1][0, 0, 0] = -5.0     # token 0 contested
    tl[1][0, 2, 0] = -3.0     # token 2 contested
    out, weight, m = apply_teacher_conflict_policy(
        routed, tl, torch.ones(1, 4), policy="mask", conflict_nats=1.0
    )
    assert torch.equal(out, routed), "mask must not alter the target itself"
    assert weight[0, 0] == 0.0 and weight[0, 2] == 0.0
    assert weight[0, 1] > 0.0 and weight[0, 3] > 0.0
    assert m["mt_opd/conflict/policy_agree_frac"] == pytest.approx(0.5)


def test_mask_rescales_so_total_loss_is_preserved() -> None:
    """Dropping tokens must not quietly shrink the effective learning rate."""
    routed = torch.randn(1, 4, 2)
    tl = _teachers(0.0, t=4, k=2)
    tl[1][0, 0, 0] = -9.0
    tl[1][0, 1, 0] = -9.0     # half the tokens contested
    mask = torch.ones(1, 4)
    _, weight, m = apply_teacher_conflict_policy(
        routed, tl, mask, policy="mask", conflict_nats=1.0
    )
    assert (weight * mask).sum().item() == pytest.approx(mask.sum().item(), rel=1e-5)
    assert m["mt_opd/conflict/mask_rescale"] == pytest.approx(2.0, rel=1e-5)


def test_mask_with_total_agreement_is_uniform() -> None:
    routed = torch.randn(1, 4, 2)
    _, weight, _ = apply_teacher_conflict_policy(
        routed, _teachers(0.0, t=4, k=2), torch.ones(1, 4),
        policy="mask", conflict_nats=1.0,
    )
    assert torch.allclose(weight, torch.ones_like(weight))


def test_consensus_averages_only_where_teachers_agree() -> None:
    routed = torch.full((1, 2, 2), 7.0)
    a = torch.zeros(1, 2, 2)
    b = torch.zeros(1, 2, 2)
    b[0, 0, 0] = -4.0     # token 0 contested -> keep routed
    b[0, 1, 0] = -0.2     # token 1 agrees   -> use mean
    out, weight, m = apply_teacher_conflict_policy(
        routed, [a, b], torch.ones(1, 2), policy="consensus", conflict_nats=1.0
    )
    assert weight is None
    assert out[0, 0, 0].item() == pytest.approx(7.0), "contested token keeps routed target"
    assert out[0, 1, 0].item() == pytest.approx(-0.1), "agreeing token uses teacher mean"
    assert m["mt_opd/conflict/consensus_frac"] == pytest.approx(0.5)


def test_consensus_with_total_conflict_keeps_routed_everywhere() -> None:
    """The degenerate case must fall back to plain MT, not to a blurred average."""
    routed = torch.full((1, 3, 2), 2.0)
    out, _, m = apply_teacher_conflict_policy(
        routed, _teachers(9.0, t=3, k=2), torch.ones(1, 3),
        policy="consensus", conflict_nats=1.0,
    )
    assert torch.equal(out, routed)
    assert m["mt_opd/conflict/policy_agree_frac"] == pytest.approx(0.0)


def test_shape_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="do not match routed"):
        apply_teacher_conflict_policy(
            torch.zeros(1, 4, 3), _teachers(1.0, t=2, k=2), torch.ones(1, 4),
            policy="mask",
        )


@pytest.mark.parametrize("policy", ["mask", "consensus"])
def test_padding_excluded_from_agreement_fraction(policy) -> None:
    routed = torch.randn(1, 4, 2)
    tl = _teachers(0.0, t=4, k=2)
    tl[1][0, 2, 0] = -8.0     # contested, but padded out
    tl[1][0, 3, 0] = -8.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    _, _, m = apply_teacher_conflict_policy(
        routed, tl, mask, policy=policy, conflict_nats=1.0
    )
    assert m["mt_opd/conflict/policy_agree_frac"] == pytest.approx(1.0), (
        "conflict in padding must not count against the agreement rate"
    )


# --- M3 quantile thresholding ------------------------------------------------
#
# Added after measuring the instrumented MT run: mean teacher spread is 0.131 nats
# and only 0.70% of tokens exceed 1.0 nats, while spread_max reaches 29.2. So the
# absolute thresholds originally chosen (0.5 / 1.0 / 2.0 nats) all sit far above the
# mean and would each contest well under 1% of tokens -- three points that would have
# produced three nulls. The quantile pins the contested fraction instead.

def test_quantile_threshold_contests_the_intended_fraction() -> None:
    routed = torch.zeros(1, 100, 2)
    a = torch.zeros(1, 100, 2)
    b = torch.zeros(1, 100, 2)
    # spread rises linearly 0..0.99 nats, so the q-th quantile contests ~(1-q).
    b[0, :, 0] = -torch.linspace(0, 0.99, 100)
    _, weight, m = apply_teacher_conflict_policy(
        routed, [a, b], torch.ones(1, 100), policy="mask", conflict_quantile=0.9
    )
    assert m["mt_opd/conflict/policy_agree_frac"] == pytest.approx(0.9, abs=0.02)
    assert m["mt_opd/conflict/policy_threshold_nats"] == pytest.approx(0.89, abs=0.03)


def test_quantile_beats_a_blind_absolute_threshold_on_real_scale() -> None:
    """The concrete failure the quantile fixes, at the measured spread scale."""
    routed = torch.zeros(1, 1000, 2)
    a = torch.zeros(1, 1000, 2)
    b = torch.zeros(1, 1000, 2)
    g = torch.Generator().manual_seed(0)
    # Heavy-tailed, mean ~0.13 nats with a few very large values, as measured.
    spread = torch.abs(torch.randn(1000, generator=g)) * 0.10
    spread[:7] = 20.0
    b[0, :, 0] = -spread

    _, _, m_abs = apply_teacher_conflict_policy(
        routed, [a, b], torch.ones(1, 1000), policy="mask", conflict_nats=1.0
    )
    _, _, m_q = apply_teacher_conflict_policy(
        routed, [a, b], torch.ones(1, 1000), policy="mask", conflict_quantile=0.8
    )
    assert m_abs["mt_opd/conflict/policy_agree_frac"] > 0.98, (
        "1.0 nats leaves almost everything uncontested at the measured scale"
    )
    assert m_q["mt_opd/conflict/policy_agree_frac"] == pytest.approx(0.8, abs=0.02), (
        "the quantile must hit the requested fraction regardless of scale"
    )


def test_quantile_overrides_the_absolute_threshold() -> None:
    routed = torch.zeros(1, 10, 2)
    a = torch.zeros(1, 10, 2)
    b = torch.zeros(1, 10, 2)
    b[0, :, 0] = -torch.linspace(0, 1.0, 10)
    _, _, m = apply_teacher_conflict_policy(
        routed, [a, b], torch.ones(1, 10),
        policy="mask", conflict_nats=99.0, conflict_quantile=0.5,
    )
    assert m["mt_opd/conflict/policy_agree_frac"] == pytest.approx(0.5, abs=0.11), (
        "conflict_nats=99 would contest nothing; the quantile must win"
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_quantile_fails_loudly(bad) -> None:
    with pytest.raises(ValueError, match="conflict_quantile must be in"):
        apply_teacher_conflict_policy(
            torch.zeros(1, 4, 2), _teachers(1.0, t=4, k=2), torch.ones(1, 4),
            policy="mask", conflict_quantile=bad,
        )


def test_quantile_ignores_padding_when_computing_the_threshold() -> None:
    """Padding spread must not shift the quantile."""
    routed = torch.zeros(1, 10, 2)
    a = torch.zeros(1, 10, 2)
    b = torch.zeros(1, 10, 2)
    b[0, :5, 0] = -torch.linspace(0, 0.4, 5)   # real tokens: small spread
    b[0, 5:, 0] = -50.0                         # padding: huge spread
    mask = torch.tensor([[1.0] * 5 + [0.0] * 5])
    _, _, m = apply_teacher_conflict_policy(
        routed, [a, b], mask, policy="mask", conflict_quantile=0.8
    )
    assert m["mt_opd/conflict/policy_threshold_nats"] < 1.0, (
        "threshold must come from real tokens only, not the padded tail"
    )

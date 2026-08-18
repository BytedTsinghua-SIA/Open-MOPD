"""ExOPD reward extrapolation.

G-OPD reframes on-policy distillation as dense KL-constrained RL whose optimum is

    log pi_theta = log pi* + (lambda - 1) (log pi* - log pi_ref)

Standard OPD is lambda == 1. The extra term is exactly Direct-OPD's reward,
``student_weights * (log pi* - log pi_ref)``, so ExOPD is a linear combination of
two rewards this codebase already computes.

The load-bearing property is that lambda == 1 must reproduce plain opd_kl
*exactly*, not approximately -- otherwise turning the feature on silently
perturbs every existing recipe.
"""
import math

import pytest
import torch

from verl.workers.actor.dp_actor import _compute_delta_opd_rm_scores, _compute_exopd_rm_scores


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_lambda_one_is_bitwise_identity() -> None:
    """The whole regression guarantee: lambda=1 returns opd_rm_scores untouched."""
    opd = _rand((2, 5, 4), seed=1)
    delta = _rand((2, 5, 4), seed=2)
    out = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=1.0)
    assert torch.equal(out, opd), "lambda=1 must not perturb the standard OPD reward at all"


def test_lambda_one_ignores_delta_entirely() -> None:
    """Even a pathological delta cannot move the reward at lambda=1."""
    opd = _rand((3, 4, 2), seed=3)
    for delta in (torch.full((3, 4, 2), 1e6), torch.zeros(3, 4, 2), torch.full((3, 4, 2), -1e6)):
        out = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=1.0)
        assert torch.equal(out, opd)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.25, 1.5, 2.0])
def test_matches_the_closed_form(lam: float) -> None:
    opd = _rand((2, 6, 3), seed=4)
    delta = _rand((2, 6, 3), seed=5)
    out = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=lam)
    torch.testing.assert_close(out, opd + (lam - 1.0) * delta)


def test_extrapolation_and_interpolation_move_opposite_ways() -> None:
    """lambda>1 must add the delta direction and lambda<1 must subtract it, so the
    sign convention matches the paper's optimum."""
    opd = torch.zeros(1, 3, 2)
    delta = torch.ones(1, 3, 2)
    up = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=1.25)
    down = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=0.75)
    assert (up > 0).all()
    assert (down < 0).all()
    torch.testing.assert_close(up, -down)


def test_output_is_detached() -> None:
    opd = _rand((1, 3, 2), seed=6).requires_grad_(True)
    delta = _rand((1, 3, 2), seed=7).requires_grad_(True)
    out = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=1.25)
    assert not out.requires_grad, "rewards must not carry gradient into the actor loss"


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        _compute_exopd_rm_scores(
            opd_rm_scores=torch.zeros(1, 3, 2),
            delta_rm_scores=torch.zeros(1, 3, 4),
            exopd_lambda=1.25,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_lambda_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        _compute_exopd_rm_scores(
            opd_rm_scores=torch.zeros(1, 2, 2),
            delta_rm_scores=torch.zeros(1, 2, 2),
            exopd_lambda=bad,
        )


def test_composes_with_the_real_delta_opd_reward() -> None:
    """End-to-end shape check against the actual Direct-OPD computation, so the
    two rewards really are combinable as claimed."""
    B, T, K = 2, 4, 3
    student = _rand((B, T, K), seed=8)
    teacher_rl = _rand((B, T, K), seed=9)
    teacher_ref = _rand((B, T, K), seed=10)

    delta_scores, delta_ratio = _compute_delta_opd_rm_scores(
        student_logp=student, teacher_rl_logp=teacher_rl, teacher_ref_logp=teacher_ref)
    assert delta_scores.shape == (B, T, K)

    # a stand-in for the standard OPD reward, same shape
    opd_scores = -(student - teacher_rl) * torch.softmax(student, dim=-1)

    out = _compute_exopd_rm_scores(
        opd_rm_scores=opd_scores, delta_rm_scores=delta_scores, exopd_lambda=1.25)
    assert out.shape == (B, T, K)
    assert torch.isfinite(out).all()
    # and the delta term is the teacher/reference log-ratio, weighted
    torch.testing.assert_close(delta_ratio, teacher_rl - teacher_ref)


def test_reference_equal_to_teacher_collapses_to_plain_opd() -> None:
    """If pi_ref == pi*, the extrapolation term vanishes for any lambda, because
    there is no RL direction to extend. A useful sanity property: it means the
    feature is inert when the teacher was never RL-trained away from its base.
    """
    B, T, K = 2, 3, 4
    student = _rand((B, T, K), seed=11)
    teacher = _rand((B, T, K), seed=12)
    delta_scores, delta_ratio = _compute_delta_opd_rm_scores(
        student_logp=student, teacher_rl_logp=teacher, teacher_ref_logp=teacher)
    assert torch.count_nonzero(delta_ratio) == 0
    opd_scores = _rand((B, T, K), seed=13)
    for lam in (0.5, 1.0, 1.25, 2.0):
        out = _compute_exopd_rm_scores(
            opd_rm_scores=opd_scores, delta_rm_scores=delta_scores, exopd_lambda=lam)
        torch.testing.assert_close(out, opd_scores)


def test_lambda_scales_the_delta_contribution_linearly() -> None:
    opd = torch.zeros(1, 4, 2)
    delta = _rand((1, 4, 2), seed=14)
    a = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=1.5)
    b = _compute_exopd_rm_scores(opd_rm_scores=opd, delta_rm_scores=delta, exopd_lambda=2.0)
    # (2.0-1) is twice (1.5-1), so the contribution must double
    torch.testing.assert_close(b, 2.0 * a)

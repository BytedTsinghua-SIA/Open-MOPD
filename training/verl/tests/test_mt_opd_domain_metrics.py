"""Per-domain instrumentation for MT-OPD.

The 2026-08-01 domain-weight sweep (2:2:1 / 3:3:1 / 2:2:0.5) came back null, totals
within 0.07 of each other, and it could not be diagnosed because the MT runs logged
nothing per domain. The reason turned out to be that under ``loss_agg_mode=token-mean``
a domain's gradient share follows its **token** share, and measured student response
lengths are math 11570 / code 9032 / IF 364 -- so IF holds 20% of prompts and 0.9% of
tokens, and moving 2:2:1 -> 3:3:1 shifts token share by 0.1-0.3pp.

These tests pin the quantities that make that visible, plus the cross-teacher conflict
signal, which is free: the MT path evaluates every teacher on every student token and
discards all but the routed one.
"""
import numpy as np
import pytest
import torch

from verl.workers.actor.mt_opd import (
    build_domain_weights,
    compute_domain_share_metrics,
    compute_teacher_conflict_metrics,
    select_routed_teacher_logprobs,
)


def test_token_share_diverges_from_prompt_share() -> None:
    """The central measurement: equal prompts, wildly unequal tokens."""
    domains = ["math", "IF"]
    mask = torch.zeros(2, 100)
    mask[0, :80] = 1.0   # math: 80 tokens
    mask[1, :4] = 1.0    # IF: 4 tokens

    m = compute_domain_share_metrics(domains=domains, response_mask=mask)

    assert m["mt_opd/domain/math/prompt_share"] == pytest.approx(0.5)
    assert m["mt_opd/domain/IF/prompt_share"] == pytest.approx(0.5)
    assert m["mt_opd/domain/math/token_share"] == pytest.approx(80 / 84)
    assert m["mt_opd/domain/IF/token_share"] == pytest.approx(4 / 84)
    assert m["mt_opd/domain/IF/token_share"] < 0.05, (
        "half the prompts must be able to hold a tiny fraction of the tokens -- "
        "that gap is the whole reason the weight sweep was null"
    )


def test_shares_sum_to_one() -> None:
    domains = ["math", "code", "if", "math"]
    mask = torch.ones(4, 10)
    m = compute_domain_share_metrics(domains=domains, response_mask=mask)
    tok = sum(v for k, v in m.items() if k.endswith("/token_share"))
    pro = sum(v for k, v in m.items() if k.endswith("/prompt_share"))
    assert tok == pytest.approx(1.0)
    assert pro == pytest.approx(1.0)


def test_mean_response_length_is_per_domain() -> None:
    domains = ["math", "math", "if"]
    mask = torch.zeros(3, 50)
    mask[0, :30] = 1.0
    mask[1, :20] = 1.0
    mask[2, :5] = 1.0
    m = compute_domain_share_metrics(domains=domains, response_mask=mask)
    assert m["mt_opd/domain/math/mean_response_length"] == pytest.approx(25.0)
    assert m["mt_opd/domain/if/mean_response_length"] == pytest.approx(5.0)
    assert m["mt_opd/domain/math/prompt_count"] == pytest.approx(2.0)


def test_reward_scale_uses_absolute_value() -> None:
    """A signed reward averaging to zero can still drive a large gradient."""
    domains = ["a", "b"]
    mask = torch.ones(2, 4)
    rm = torch.zeros(2, 4)
    rm[0] = torch.tensor([5.0, -5.0, 5.0, -5.0])   # mean 0, magnitude 5
    rm[1] = torch.tensor([0.1, 0.1, 0.1, 0.1])     # mean 0.1, magnitude 0.1
    m = compute_domain_share_metrics(domains=domains, response_mask=mask, rm_scores=rm)
    assert m["mt_opd/domain/a/reward_abs_mean"] == pytest.approx(5.0)
    assert m["mt_opd/domain/b/reward_abs_mean"] == pytest.approx(0.1)


def test_reward_scale_accepts_3d_dense_rewards() -> None:
    domains = ["a"]
    mask = torch.ones(1, 3)
    rm = torch.full((1, 3, 4), 0.25)      # sums to 1.0 per token over K
    m = compute_domain_share_metrics(domains=domains, response_mask=mask, rm_scores=rm)
    assert m["mt_opd/domain/a/reward_abs_mean"] == pytest.approx(1.0)


def test_padding_is_excluded_from_shares() -> None:
    domains = ["a", "b"]
    mask = torch.zeros(2, 10)
    mask[0, :5] = 1.0
    mask[1, :5] = 1.0
    rm = torch.ones(2, 10) * 3.0          # nonzero in padding too
    m = compute_domain_share_metrics(domains=domains, response_mask=mask, rm_scores=rm)
    assert m["mt_opd/domain/a/token_share"] == pytest.approx(0.5)
    assert m["mt_opd/domain/a/reward_abs_mean"] == pytest.approx(3.0), (
        "padded positions must not dilute the per-token reward scale"
    )


def test_no_domains_yields_no_metrics() -> None:
    assert compute_domain_share_metrics(None, torch.ones(2, 3)) == {}
    assert compute_domain_share_metrics([], torch.ones(0, 3)) == {}


def test_mismatched_batch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_domain_share_metrics(["a", "b"], torch.ones(3, 4))


# --- teacher conflict -------------------------------------------------------

def test_identical_teachers_have_zero_conflict() -> None:
    t = torch.randn(2, 5, 4)
    m = compute_teacher_conflict_metrics([t, t.clone()], torch.ones(2, 5))
    assert m["mt_opd/conflict/spread_mean"] == pytest.approx(0.0)
    assert m["mt_opd/conflict/contested_frac"] == pytest.approx(0.0)


def test_conflict_measures_top1_spread_in_nats() -> None:
    a = torch.zeros(1, 2, 3)
    b = torch.zeros(1, 2, 3)
    a[0, :, 0] = -1.0
    b[0, :, 0] = -3.5        # 2.5 nats apart on the student's top-1
    m = compute_teacher_conflict_metrics([a, b], torch.ones(1, 2))
    assert m["mt_opd/conflict/spread_mean"] == pytest.approx(2.5)
    assert m["mt_opd/conflict/contested_frac"] == pytest.approx(1.0)


def test_conflict_threshold_partitions_tokens() -> None:
    a = torch.zeros(1, 4, 2)
    b = torch.zeros(1, 4, 2)
    b[0, 0, 0] = -2.0        # contested
    b[0, 1, 0] = -0.2        # agrees
    b[0, 2, 0] = -3.0        # contested
    b[0, 3, 0] = -0.1        # agrees
    m = compute_teacher_conflict_metrics([a, b], torch.ones(1, 4), conflict_nats=1.0)
    assert m["mt_opd/conflict/contested_frac"] == pytest.approx(0.5)


def test_conflict_is_reported_per_domain() -> None:
    a = torch.zeros(2, 3, 2)
    b = torch.zeros(2, 3, 2)
    b[0, :, 0] = -4.0        # math sample: heavy conflict
    b[1, :, 0] = 0.0         # if sample: none
    m = compute_teacher_conflict_metrics(
        [a, b], torch.ones(2, 3), domains=np.array(["math", "if"])
    )
    assert m["mt_opd/conflict/math/spread_mean"] == pytest.approx(4.0)
    assert m["mt_opd/conflict/if/spread_mean"] == pytest.approx(0.0)
    assert m["mt_opd/conflict/math/contested_frac"] == pytest.approx(1.0)
    assert m["mt_opd/conflict/if/contested_frac"] == pytest.approx(0.0)


def test_single_teacher_has_no_conflict_metrics() -> None:
    """Disagreement is undefined for one teacher; return nothing rather than zeros."""
    assert compute_teacher_conflict_metrics([torch.randn(1, 2, 3)], torch.ones(1, 2)) == {}
    assert compute_teacher_conflict_metrics([], torch.ones(1, 2)) == {}


def test_routing_does_not_alias_the_first_teacher() -> None:
    """The bug that would silently report zero conflict.

    The trainer holds ``teacher_logps`` as a list whose element 0 is the tensor from
    ``batch["teacher_on_student_log_probs"]``, then overwrites that same batch key
    with the routed result. If routing mutated in place, element 0 would become the
    routed tensor and every conflict measurement would compare a teacher to itself.
    """
    t0 = torch.zeros(2, 3, 2)
    t1 = torch.full((2, 3, 2), -5.0)
    teacher_logps = [t0, t1]
    before = t0.clone()

    weights = build_domain_weights(
        ["math", "if"], domain_order=["math", "if"], device="cpu"
    )
    routed = select_routed_teacher_logprobs(teacher_logps, weights)

    assert torch.equal(teacher_logps[0], before), "teacher 0 must be untouched"
    assert routed is not teacher_logps[0]
    # sample 1 routed to teacher "if", so it must carry t1's value
    assert routed[1, 0, 0].item() == pytest.approx(-5.0)

    m = compute_teacher_conflict_metrics(teacher_logps, torch.ones(2, 3))
    assert m["mt_opd/conflict/spread_mean"] == pytest.approx(5.0), (
        "conflict must be computed against the original teachers, not the routed mix"
    )

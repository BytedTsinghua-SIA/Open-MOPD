"""Hybrid RL+OPD advantage under dense top-k rewards.

``token_reward_direct_plus_grpo`` combines the dense teacher reward with a
verifiable outcome reward. Its docstring declares ``token_level_rewards: (bs,
response_length)`` and it was only ever exercised that way; with
``log_prob_top_k > 0`` the direct advantage is (bs, seq, K) and every hybrid job
crash-looped 29-30 times on

    RuntimeError: The size of tensor a (16) must match the size of tensor b (16384)

Broadcasting the outcome term over K is not the fix. The 3D policy loss sums over
K and ``ratio[b,t,k]`` is candidate k's importance ratio, so a broadcast term
would be counted K times and would credit candidates the rollout never sampled.
It belongs on the sampled candidate's slot, where the ratio is the sampled
action's and the ordinary PPO term is recovered.
"""
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    _scatter_outcome_adv_onto_sampled_topk,
    compute_token_reward_direct_plus_grpo_advantage,
)


class _Cfg:
    def __init__(self, weight=1.0, norm=True):
        self.grpo_outcome_weight = weight
        self.norm_adv_by_std_in_grpo = norm


def _scatter(grpo_adv, direct_adv, ids, responses, response_mask=None):
    if response_mask is None:
        response_mask = torch.ones(direct_adv.shape[:2])
    return _scatter_outcome_adv_onto_sampled_topk(
        grpo_adv=grpo_adv,
        direct_adv=direct_adv,
        response_mask=response_mask,
        student_top_k_ids=ids,
        responses=responses,
    )


def test_outcome_lands_only_on_the_sampled_candidate() -> None:
    direct = torch.zeros(1, 2, 4)
    ids = torch.tensor([[[7, 3, 9, 1], [2, 8, 5, 4]]])
    responses = torch.tensor([[9, 8]])
    grpo = torch.full((1, 2), 2.0)

    out, _ = _scatter(grpo, direct, ids, responses)

    assert out[0, 0, 2] == pytest.approx(2.0), "token 9 sits at k=2"
    assert out[0, 1, 1] == pytest.approx(2.0), "token 8 sits at k=1"
    assert out.sum() == pytest.approx(4.0), "exactly one slot per position carries it"


def test_total_outcome_mass_is_not_multiplied_by_k() -> None:
    """The bug a naive broadcast would introduce, stated as a test."""
    bs, seq, k = 2, 3, 8
    direct = torch.zeros(bs, seq, k)
    ids = torch.arange(bs * seq * k).reshape(bs, seq, k)
    responses = ids[:, :, 0]  # always sample the top candidate
    grpo = torch.full((bs, seq), 1.5)

    out, _ = _scatter(grpo, direct, ids, responses)

    naive = grpo.unsqueeze(-1).expand(bs, seq, k)
    assert out.sum() == pytest.approx(bs * seq * 1.5)
    assert naive.sum() == pytest.approx(bs * seq * 1.5 * k)
    assert out.sum() * k == pytest.approx(naive.sum()), (
        "broadcasting would inflate the outcome term exactly K-fold"
    )


def test_token_outside_topk_gets_no_outcome_signal() -> None:
    direct = torch.zeros(1, 2, 3)
    ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    responses = torch.tensor([[99, 5]])  # first token was not in top-k
    grpo = torch.full((1, 2), 1.0)

    out, frac = _scatter(grpo, direct, ids, responses)

    assert out[0, 0].sum() == pytest.approx(0.0)
    assert out[0, 1, 1] == pytest.approx(1.0)
    assert frac[0] == pytest.approx(0.5), "coverage must report the miss"


def test_coverage_respects_the_response_mask() -> None:
    """Padding must not count as a miss and drag the reported rate down."""
    direct = torch.zeros(1, 4, 2)
    ids = torch.tensor([[[1, 2], [3, 4], [0, 0], [0, 0]]])
    responses = torch.tensor([[1, 3, 0, 0]])
    grpo = torch.ones(1, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    _, frac = _scatter(grpo, direct, ids, responses, response_mask=mask)
    assert frac[0] == pytest.approx(1.0), "both real tokens hit; padding is excluded"


def test_negative_outcome_advantage_is_preserved() -> None:
    direct = torch.zeros(1, 1, 3)
    ids = torch.tensor([[[5, 6, 7]]])
    responses = torch.tensor([[6]])
    out, _ = _scatter(torch.full((1, 1), -3.0), direct, ids, responses)
    assert out[0, 0, 1] == pytest.approx(-3.0)
    assert out[0, 0, 0] == pytest.approx(0.0) and out[0, 0, 2] == pytest.approx(0.0)


def test_direct_advantage_is_left_intact() -> None:
    direct = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
    responses = torch.tensor([[2, 7, 12]])
    out, _ = _scatter(torch.zeros(1, 3), direct, ids, responses)
    assert torch.equal(out, torch.zeros_like(direct)), (
        "a zero outcome advantage must contribute nothing anywhere"
    )


@pytest.mark.parametrize("bad", ["ids", "responses"])
def test_missing_inputs_fail_loudly(bad: str) -> None:
    direct = torch.zeros(1, 2, 3)
    ids = torch.zeros(1, 2, 3, dtype=torch.long)
    responses = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="locate the sampled candidate"):
        _scatter(
            torch.zeros(1, 2),
            direct,
            None if bad == "ids" else ids,
            None if bad == "responses" else responses,
        )


def test_mismatched_k_fails_loudly() -> None:
    with pytest.raises(ValueError, match="K="):
        _scatter(
            torch.zeros(1, 2),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 8, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
        )


def test_mismatched_sequence_fails_loudly() -> None:
    with pytest.raises(ValueError, match="does not align"):
        _scatter(
            torch.zeros(1, 2),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 5, 4, dtype=torch.long),
            torch.zeros(1, 5, dtype=torch.long),
        )


def test_end_to_end_3d_reward_no_longer_raises() -> None:
    """The exact crash: 3D dense rewards through the hybrid estimator."""
    import numpy as np

    bs, seq, k = 4, 6, 3
    rewards = torch.randn(bs, seq, k)
    response_mask = torch.ones(bs, seq)
    ids = torch.arange(bs * seq * k).reshape(bs, seq, k)
    responses = ids[:, :, 1]
    true_reward = torch.zeros(bs, seq)
    true_reward[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    index = np.array(["g0", "g0", "g1", "g1"])

    adv, ret, extra = compute_token_reward_direct_plus_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        config=_Cfg(weight=1.0),
        true_reward_score=true_reward,
        student_top_k_ids=ids,
        responses=responses,
    )

    assert adv.shape == (bs, seq, k)
    assert torch.equal(adv, ret)
    assert extra["grpo_outcome_sampled_frac"].shape == (bs,)
    assert torch.allclose(extra["grpo_outcome_sampled_frac"], torch.ones(bs))
    # Only k=1 may differ from the pure direct advantage.
    direct = extra["token_level_advantage_direct"]
    for j in (0, 2):
        assert torch.allclose(adv[:, :, j], direct[:, :, j])
    assert not torch.allclose(adv[:, :, 1], direct[:, :, 1])


def test_weight_scales_only_the_outcome_term() -> None:
    import numpy as np

    bs, seq, k = 2, 3, 3
    rewards = torch.randn(bs, seq, k)
    response_mask = torch.ones(bs, seq)
    ids = torch.arange(bs * seq * k).reshape(bs, seq, k)
    responses = ids[:, :, 0]
    true_reward = torch.zeros(bs, seq)
    true_reward[:, -1] = torch.tensor([1.0, 0.0])
    index = np.array(["g", "g"])

    kw = dict(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        true_reward_score=true_reward,
        student_top_k_ids=ids,
        responses=responses,
    )
    full, _, ex = compute_token_reward_direct_plus_grpo_advantage(config=_Cfg(1.0), **kw)
    half, _, _ = compute_token_reward_direct_plus_grpo_advantage(config=_Cfg(0.5), **kw)
    direct = ex["token_level_advantage_direct"]

    assert torch.allclose(full - direct, 2.0 * (half - direct), atol=1e-6)


def test_2d_rewards_still_take_the_original_path() -> None:
    """Recipes without dense top-k must be untouched by this change."""
    import numpy as np

    bs, seq = 3, 4
    rewards = torch.randn(bs, seq)
    response_mask = torch.ones(bs, seq)
    true_reward = torch.zeros(bs, seq)
    true_reward[:, -1] = torch.tensor([1.0, 0.0, 1.0])
    index = np.array(["a", "a", "a"])

    adv, ret, extra = compute_token_reward_direct_plus_grpo_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        config=_Cfg(1.0),
        true_reward_score=true_reward,
    )
    assert adv.shape == (bs, seq)
    assert "grpo_outcome_sampled_frac" not in extra, (
        "the 2D path needs no sampled-candidate lookup"
    )

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import (
    _compute_delta_opd_metrics,
    _compute_ref_teacher_overlap_metrics,
    _compute_student_weighted_teacher_logprob_metrics,
    _pop_direct_opd_rollout_options,
)
from verl.workers.actor.dp_actor import _compute_delta_opd_rm_scores
from verl.workers.fsdp_workers import RewardModelWorker, _compute_student_topk_in_teacher_top_p_mask


def test_direct_opd_reward_uses_detached_student_topk_weights():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -1.0, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
    )

    expected_delta = teacher_rl_logp - teacher_ref_logp
    expected_weights = torch.softmax(student_logp.detach(), dim=-1)

    assert rm_scores.shape == torch.Size([1, 1, 3])
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert not rm_scores.requires_grad
    assert not delta.requires_grad


def test_direct_opd_reward_masks_invalid_candidates_before_weighting():
    student_logp = torch.tensor([[[-0.1, -2.0, -3.0]]], requires_grad=True)
    teacher_rl_logp = torch.tensor([[[-0.2, -torch.inf, -4.0]]])
    teacher_ref_logp = torch.tensor([[[-0.5, -2.0, -2.5]]])
    valid_mask = torch.tensor([[[True, False, True]]])

    rm_scores, delta = _compute_delta_opd_rm_scores(
        student_logp=student_logp,
        teacher_rl_logp=teacher_rl_logp,
        teacher_ref_logp=teacher_ref_logp,
        valid_mask=valid_mask,
    )

    expected_delta = (teacher_rl_logp - teacher_ref_logp).masked_fill(~valid_mask, 0.0)
    masked_student_logp = student_logp.detach().masked_fill(~valid_mask, -torch.inf)
    expected_weights = torch.softmax(masked_student_logp, dim=-1)
    expected_weights = torch.nan_to_num(expected_weights, nan=0.0, posinf=0.0, neginf=0.0)

    assert torch.isfinite(rm_scores).all()
    assert torch.isfinite(delta).all()
    assert torch.allclose(delta, expected_delta)
    assert torch.allclose(rm_scores, expected_weights * expected_delta)
    assert rm_scores[0, 0, 1].item() == 0.0
    assert not rm_scores.requires_grad


def test_direct_opd_reward_rejects_mismatched_shapes():
    student_logp = torch.zeros(2, 3, 4)
    teacher_rl_logp = torch.zeros(2, 3, 4)
    teacher_ref_logp = torch.zeros(2, 3, 5)

    try:
        _compute_delta_opd_rm_scores(
            student_logp=student_logp,
            teacher_rl_logp=teacher_rl_logp,
            teacher_ref_logp=teacher_ref_logp,
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched Direct-OPD shapes to raise ValueError")


def test_student_topk_teacher_top_p_mask_includes_threshold_crossing_token():
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    student_ids = torch.tensor([[0, 1, 2, 3]])

    mask = _compute_student_topk_in_teacher_top_p_mask(logits, student_ids, top_p=0.70)

    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.equal(mask, expected)


def test_teacher_top_p_intersec_strategy_returns_top_p_overlap_mask():
    logits = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    student_ids = torch.tensor([[0, 1, 2, 3]])

    (
        teacher_on_student_log_probs,
        valid_counts,
        overlap_mask,
        teacher_top_k_ids,
        teacher_top_k_log_probs,
        teacher_in_student_mask,
    ) = RewardModelWorker._compute_teacher_top_k_log_probs(
        None,
        logits=logits,
        student_ids=student_ids,
        top_k=2,
        strategy="top_p_intersec",
        top_p_intersec_p=0.70,
    )

    expected_log_probs = torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))
    expected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    assert torch.allclose(teacher_on_student_log_probs, expected_log_probs)
    assert torch.equal(valid_counts, torch.tensor([2]))
    assert torch.equal(overlap_mask, expected_mask)
    assert torch.equal(teacher_top_k_ids, torch.tensor([[0, 1]]))
    assert torch.allclose(teacher_top_k_log_probs, torch.log(torch.tensor([[0.50, 0.30]])))
    assert torch.equal(teacher_in_student_mask, torch.tensor([[1.0, 1.0]]))


def test_direct_opd_rollout_options_are_removed_before_rollout_config_instantiation():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "reward_mode": "delta_opd",
                    "top_k_strategy": "top_p_intersec",
                    "top_p_intersec_p": 0.85,
                }
            }
        }
    )

    options = _pop_direct_opd_rollout_options(config)

    assert options == {"reward_mode": "delta_opd"}
    assert "reward_mode" not in config.actor_rollout_ref.rollout
    assert config.actor_rollout_ref.rollout.top_k_strategy == "top_p_intersec"
    assert config.actor_rollout_ref.rollout.top_p_intersec_p == 0.85


def test_direct_opd_metrics_include_weighted_pos_frac_and_teacher_gap():
    student_topk_log_probs = torch.log(torch.tensor([[[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]]))
    teacher_ref_on_student = torch.log(torch.tensor([[[0.3, 0.3, 0.4], [0.1, 0.7, 0.2]]]))
    response_mask = torch.tensor([[1, 1]])
    delta = teacher_on_student - teacher_ref_on_student
    rm_scores = torch.softmax(student_topk_log_probs, dim=-1) * delta

    metrics = {}
    metrics.update(
        _compute_delta_opd_metrics(
            delta=delta,
            rm_scores=rm_scores,
            response_mask=response_mask,
            student_topk_log_probs=student_topk_log_probs,
        )
    )
    metrics.update(
        _compute_student_weighted_teacher_logprob_metrics(
            student_topk_log_probs=student_topk_log_probs,
            teacher_on_student_log_probs=teacher_on_student,
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            response_mask=response_mask,
        )
    )
    metrics.update(
        _compute_ref_teacher_overlap_metrics(
            teacher_ref_on_student_log_probs=teacher_ref_on_student,
            teacher_ref_overlap_mask=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
            response_mask=response_mask,
        )
    )

    assert "delta_opd/weighted_reward_mean" in metrics
    assert "delta_opd/topk_log_ratio_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_pos_frac" in metrics
    assert "delta_opd/student_weighted_teacher_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_ref_logprob" in metrics
    assert "delta_opd/student_weighted_teacher_logprob_gap" in metrics
    assert "delta_opd/ref_teacher_overlap_ratio" in metrics

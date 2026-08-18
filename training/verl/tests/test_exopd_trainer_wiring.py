"""ExOPD trainer wiring: the standard OPD reward must be computed, not assumed.

These tests cover the defect class that ``test_exopd_reward.py`` cannot see. That
file tests the reward *math* in isolation and passed while every real ExOPD job
crash-looped 18-19 times on ``KeyError: 'rm_scores'``.

The cause was an assumption about where the standard OPD reward comes from. For
``log_prob_top_k > 0`` the reward-model pass deliberately returns no reward --
``fsdp_workers`` sets ``rm_scores = None`` and defers the computation to the
trainer's ``elif top_k > 0`` branch. ExOPD takes an earlier branch of that same
if/elif chain, so that branch is unreachable and the reward is never produced.

So the invariant worth pinning is not "lambda=1 is identity" (already covered)
but "the ExOPD path itself produces the standard OPD reward before combining".
"""
import re
from pathlib import Path

import pytest
import torch

from verl.protocol import DataProto

_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "verl" / "trainer" / "ppo" / "ray_trainer.py"
_WORKERS = _REPO / "verl" / "workers" / "fsdp_workers.py"


def test_rm_score_pass_yields_no_reward_for_topk() -> None:
    """The precondition ExOPD got wrong, read off the source that establishes it.

    If this ever flips -- if the reward-model pass starts returning rm_scores for
    top_k > 0 -- then the trainer's extra distillation call becomes redundant
    work and this test should fail loudly rather than let it linger.
    """
    src = _WORKERS.read_text()
    guard = re.search(
        r"if top_k > 0:\s*\n\s*# Reward calculation is moved to ray_trainer[^\n]*\n"
        r"(?:\s*#[^\n]*\n)*\s*rm_scores = None",
        src,
    )
    assert guard is not None, (
        "fsdp_workers no longer defers reward computation to ray_trainer for "
        "top_k>0; ExOPD's assumption about where the standard OPD reward comes "
        "from must be rechecked"
    )


def test_exopd_branch_computes_the_standard_reward_before_combining() -> None:
    """The ExOPD branch must call distillation and stash it under opd_rm_scores."""
    src = _TRAINER.read_text()
    start = src.index('if reward_mode == "exopd":')
    end = src.index('with marked_timer("compute_teacher_ref_rm_score"', start)
    branch = src[start:end]

    assert "compute_distillation_reward" in branch, (
        "the ExOPD branch must compute the standard OPD reward itself: the "
        "`elif top_k > 0` branch that normally does so is unreachable from here"
    )
    assert "opd_rm_scores" in branch, "the standard reward must be stashed under opd_rm_scores"

    combine = src.index("compute_exopd_reward(batch)", start)
    assert src.index("compute_distillation_reward", start) < combine, (
        "distillation must run before the rewards are combined"
    )


def test_union_rejects_two_different_rm_scores() -> None:
    """Why the rename is load-bearing rather than cosmetic.

    Both the standard and extrapolated rewards are returned under the key
    ``rm_scores``. DataProto.union asserts equality on duplicate keys, so
    unioning both without renaming raises -- it does not silently overwrite.
    """
    base = DataProto.from_dict(tensors={"rm_scores": torch.zeros(2, 3)})
    other = DataProto.from_dict(tensors={"rm_scores": torch.ones(2, 3)})
    with pytest.raises(AssertionError):
        base.union(other)


def test_renaming_lets_both_rewards_coexist() -> None:
    """The trainer's actual fix: rename, then union, then combine."""
    standard = torch.full((2, 3, 4), 0.5)
    distill_out = DataProto.from_dict(tensors={"rm_scores": standard})

    renamed = {k: v for k, v in distill_out.batch.items() if k != "rm_scores"}
    renamed["opd_rm_scores"] = distill_out.batch["rm_scores"]
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones(2, 3)})
    batch = batch.union(DataProto.from_dict(tensors=renamed))

    assert "opd_rm_scores" in batch.batch.keys()
    assert "rm_scores" not in batch.batch.keys(), (
        "rm_scores must stay free for the extrapolated reward to claim"
    )

    extrapolated = DataProto.from_dict(tensors={"rm_scores": torch.full((2, 3, 4), 0.9)})
    batch = batch.union(extrapolated)
    assert torch.equal(batch.batch["opd_rm_scores"], standard)
    assert torch.equal(batch.batch["rm_scores"], torch.full((2, 3, 4), 0.9))


def test_extra_distillation_keys_survive_the_rename() -> None:
    """Non-only_stu strategies return extra tensors; they must not be dropped.

    ``union`` and ``union-intersection`` also return union_top_k_ids and
    student_log_probs_on_teacher_ids. The rename must preserve them, since a
    later metrics block reads them.
    """
    distill_out = DataProto.from_dict(tensors={
        "rm_scores": torch.zeros(2, 3, 4),
        "union_top_k_ids": torch.ones(2, 3, 8, dtype=torch.long),
        "student_log_probs_on_teacher_ids": torch.full((2, 3, 4), -1.0),
    })
    renamed = {k: v for k, v in distill_out.batch.items() if k != "rm_scores"}
    renamed["opd_rm_scores"] = distill_out.batch["rm_scores"]

    assert set(renamed) == {
        "opd_rm_scores", "union_top_k_ids", "student_log_probs_on_teacher_ids",
    }

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _ensure_validation_data_source,
    _validation_batch_uses_model_reward,
    _validation_ground_truth,
)
from verl.workers.actor.mt_opd import build_domain_weights, select_routed_teacher_logprobs


def _training_batch() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1], [0, 1]]),
        },
        non_tensors={
            "data_source": np.array(["math", "code"], dtype=object),
            "reward_model": np.array(
                [
                    {"style": "rule", "ground_truth": "1"},
                    {"style": "rule", "ground_truth": "2"},
                ],
                dtype=object,
            ),
            "extra_info": np.array([{"domain": "math"}, {"domain": "code"}], dtype=object),
            "uid": np.array(["uid-0", "uid-1"], dtype=object),
            "domain": np.array(["math", "code"], dtype=object),
            "raw_prompt": np.array(
                [
                    [{"role": "user", "content": "math"}],
                    [{"role": "user", "content": "code"}],
                ],
                dtype=object,
            ),
        },
    )


@pytest.mark.parametrize("async_rollout_mode", [False, True])
def test_get_gen_batch_preserves_mt_opd_domain_for_teacher_routing(async_rollout_mode: bool) -> None:
    trainer = object.__new__(RayPPOTrainer)
    trainer.async_rollout_mode = async_rollout_mode
    batch = _training_batch()

    gen_batch = trainer._get_gen_batch(batch)

    assert batch.non_tensor_batch["domain"].tolist() == ["math", "code"]
    assert batch.non_tensor_batch["uid"].tolist() == ["uid-0", "uid-1"]
    if async_rollout_mode:
        assert gen_batch.non_tensor_batch["domain"].tolist() == ["math", "code"]
    else:
        assert "domain" not in gen_batch.non_tensor_batch


def test_formal_eval_dataset_column_becomes_validation_data_source() -> None:
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2]])},
        non_tensors={"dataset": np.array(["livecodebench_v6"], dtype=object)},
    )

    _ensure_validation_data_source(batch)

    assert batch.non_tensor_batch["data_source"].tolist() == ["livecodebench_v6"]


def test_mixed_validation_batch_backfills_only_null_data_sources() -> None:
    """MT-OPD concatenates math parquets (which carry ``data_source``) with the
    aligned Code/IF parquets (which do not).  ``concatenate_datasets`` unions the
    schemas and fills the absent column with ``None``, so the column exists but
    is null for exactly the Code/IF rows.  Those rows must still be routed.
    """
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.arange(10).reshape(5, 2)},
        non_tensors={
            "dataset": np.array(
                ["aime24", "livecodebench_v5", "ifeval", "ifbench_test", "aime25"],
                dtype=object,
            ),
            "data_source": np.array(["aime24", None, None, "", "aime25"], dtype=object),
        },
    )

    _ensure_validation_data_source(batch)

    assert batch.non_tensor_batch["data_source"].tolist() == [
        "aime24",
        "livecodebench_v5",
        "ifeval",
        "ifbench_test",
        "aime25",
    ]


def test_existing_rl_data_sources_are_never_overwritten() -> None:
    """RL datasets legitimately use a ``data_source`` that differs from any
    ``dataset`` label; a fully populated column must pass through untouched.
    """
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.arange(4).reshape(2, 2)},
        non_tensors={
            "dataset": np.array(["ignored_a", "ignored_b"], dtype=object),
            "data_source": np.array(["math_dapo_boxed", "nemotron_if_rl"], dtype=object),
        },
    )

    _ensure_validation_data_source(batch)

    assert batch.non_tensor_batch["data_source"].tolist() == [
        "math_dapo_boxed",
        "nemotron_if_rl",
    ]


def test_repaired_data_sources_restore_per_domain_validation_repeat_counts() -> None:
    """The repair also fixes rollout profile selection, not just reward routing.

    ``build_validation_repeat_plan`` keys off the same ``data_source`` values, so
    null Code rows silently fell back to the default profile (``n=1``) instead of
    LCB's ``n=10``.  That is why an MT-OPD validation round emitted only 342 Code
    rollouts for 342 Code prompts.
    """
    from verl.trainer.ppo.validation_sampling import build_validation_repeat_plan

    batch = DataProto.from_dict(
        tensors={"input_ids": torch.arange(6).reshape(3, 2)},
        non_tensors={
            "dataset": np.array(["aime24", "livecodebench_v5", "ifeval"], dtype=object),
            "data_source": np.array(["aime24", None, None], dtype=object),
        },
    )

    _ensure_validation_data_source(batch)
    indices, _, resolved = build_validation_repeat_plan(
        batch.non_tensor_batch["data_source"],
        {"n": 1, "temperature": 1.0},
        {
            "aime24": {"n": 64, "temperature": 0.6},
            "livecodebench_v5": {"n": 10, "temperature": 1.0},
            "ifeval": {"n": 1, "temperature": 1.0},
        },
    )

    assert indices.count(0) == 64
    assert indices.count(1) == 10
    assert indices.count(2) == 1
    assert set(resolved) == {"aime24", "livecodebench_v5", "ifeval"}
    assert "" not in resolved


def test_validation_data_source_repair_is_noop_without_dataset_column() -> None:
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2]])},
        non_tensors={"data_source": np.array(["aime24"], dtype=object)},
    )

    _ensure_validation_data_source(batch)

    assert batch.non_tensor_batch["data_source"].tolist() == ["aime24"]
    assert "dataset" not in batch.non_tensor_batch


@pytest.mark.parametrize(
    ("reward_model", "expected"),
    [
        (None, False),
        ({"style": "rule", "ground_truth": "1"}, False),
        ({"style": "model"}, True),
    ],
)
def test_validation_model_reward_detection_is_optional(reward_model, expected: bool) -> None:
    non_tensors = {"dataset": np.array(["formal_eval"], dtype=object)}
    if reward_model is not None:
        non_tensors["reward_model"] = np.array([reward_model], dtype=object)
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2]])},
        non_tensors=non_tensors,
    )

    assert _validation_batch_uses_model_reward(batch) is expected


@pytest.mark.parametrize(
    ("reward_model", "expected"),
    [
        (None, None),
        ({"style": "rule", "ground_truth": "42"}, "42"),
        ({"style": "model"}, None),
    ],
)
def test_validation_ground_truth_is_optional(reward_model, expected) -> None:
    non_tensors = {"dataset": np.array(["formal_eval"], dtype=object)}
    if reward_model is not None:
        values = np.empty(1, dtype=object)
        values[0] = reward_model
        non_tensors["reward_model"] = values
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2]])},
        non_tensors=non_tensors,
    )

    assert _validation_ground_truth(batch[0]) == expected


def test_mt_opd_domain_survives_async_rollout_union_and_repeat() -> None:
    trainer = object.__new__(RayPPOTrainer)
    trainer.async_rollout_mode = True
    batch = _training_batch()
    gen_batch = trainer._get_gen_batch(batch)

    # The async agent loop returns request metadata with each generated row.
    rollout = DataProto.from_dict(
        tensors={"responses": torch.tensor([[5], [6]])},
        non_tensors={"domain": gen_batch.non_tensor_batch["domain"].copy()},
    )
    batch = batch.repeat(repeat_times=1, interleave=True).union(rollout)

    assert batch.non_tensor_batch["domain"].tolist() == ["math", "code"]

    weights = build_domain_weights(
        domains=batch.non_tensor_batch["domain"],
        domain_order=["math", "code", "if"],
        device="cpu",
    )
    teachers = [
        torch.full((2, 1, 1), 10.0),
        torch.full((2, 1, 1), 20.0),
        torch.full((2, 1, 1), 30.0),
    ]
    routed = select_routed_teacher_logprobs(teachers, weights)

    assert routed[:, 0, 0].tolist() == [10.0, 20.0]

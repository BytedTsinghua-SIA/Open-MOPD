import sys
from pathlib import Path

from omegaconf import OmegaConf


VERL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERL_ROOT))

from verl.trainer.ppo.validation_sampling import (  # noqa: E402
    build_validation_repeat_plan,
    resolve_validation_profile,
)


def test_mixed_validation_suite_uses_rl_aligned_repeat_counts_and_params():
    defaults = {
        "n": 1,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": 30_000,
    }
    overrides = {
        "aime24": {"n": 64, "temperature": 0.6},
        "livecodebench_v5": {"n": 10, "stop_token_ids": [151645, 151643]},
        "ifeval": {
            "n": 1,
            "max_tokens": 10_000,
            "stop_token_ids": [151645, 151643],
        },
    }

    indices, request_params, resolved = build_validation_repeat_plan(
        ["aime24", "livecodebench_v5", "ifeval"], defaults, overrides
    )

    assert indices == [0] * 64 + [1] * 10 + [2]
    assert len(request_params) == 75
    assert request_params[0] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": 30_000,
    }
    assert request_params[64]["stop_token_ids"] == [151645, 151643]
    assert request_params[-1]["max_tokens"] == 10_000
    assert resolved["aime24"]["n"] == 64
    assert resolved["livecodebench_v5"]["n"] == 10
    assert resolved["ifeval"]["n"] == 1
    assert all("n" not in params for params in request_params)


def test_greedy_source_override_forces_zero_temperature():
    profile = resolve_validation_profile(
        "greedy_eval",
        {"n": 2, "do_sample": True, "temperature": 0.7},
        {"greedy_eval": {"do_sample": False}},
    )

    assert profile["n"] == 2
    assert profile["temperature"] == 0.0


def test_hydra_stop_token_ids_are_normalized_to_python_list():
    profiles = OmegaConf.create(
        {"livecodebench_v5": {"n": 1, "stop_token_ids": [151645, 151643]}}
    )

    _, request_params, _ = build_validation_repeat_plan(
        ["livecodebench_v5"], {"n": 1}, profiles
    )

    assert type(request_params[0]["stop_token_ids"]) is list
    assert request_params[0]["stop_token_ids"] == [151645, 151643]


def test_non_positive_repeat_count_is_rejected():
    try:
        resolve_validation_profile("broken", {"n": 1}, {"broken": {"n": 0}})
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("expected a non-positive validation repeat count to fail")

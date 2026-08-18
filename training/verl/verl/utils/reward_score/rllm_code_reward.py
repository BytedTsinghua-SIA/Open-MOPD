from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

VENDOR_ROOT = Path(__file__).resolve().parent / "rllm_vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from rllm.rewards.code_reward import RewardCodeFn  # noqa: E402
from rllm.rewards.reward_types import RewardConfig, RewardType  # noqa: E402

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n.*?```", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_FORMAT_WEIGHT = 0.2
ANSWER_WEIGHT = 0.8
DATA_SOURCE_ALIASES = {
    "livecodebench_v6_openopd": "livecodebench",
    "livecodebench_v5": "livecodebench",
    "livecodebench_v6": "livecodebench",
}


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@lru_cache(maxsize=16)
def _reward_fn(max_tests: int, timeout: int, testcase_max_workers: int) -> RewardCodeFn:
    return RewardCodeFn(
        RewardConfig(
            code_max_tests=max_tests,
            code_timeout=timeout,
            code_testcase_max_workers=testcase_max_workers,
        )
    )


def _load_ground_truth(ground_truth: Any) -> Any:
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            return ground_truth
    return ground_truth


def _ensure_code_block(solution: str) -> str:
    if CODE_BLOCK_RE.search(solution):
        return solution
    return f"```python\n{solution}\n```"


def _has_closed_think_block(solution: str) -> bool:
    return THINK_BLOCK_RE.search(str(solution or "")) is not None


def _with_think_format_score(result: float | dict[str, Any], solution_str: str) -> dict[str, Any]:
    if isinstance(result, dict):
        output = dict(result)
        task_score = float(output.get("acc", output.get("score", 0.0)) or 0.0)
    else:
        output = {}
        task_score = float(result)
    think_format_score = float(_has_closed_think_block(solution_str))
    think_score = THINK_FORMAT_WEIGHT * think_format_score
    answer_score = ANSWER_WEIGHT * task_score
    final_score = think_score + answer_score
    output.update(
        {
            "score": final_score,
            "think_score": think_score,
            "answer_score": answer_score,
            "final_score": final_score,
            "base_score": task_score,
            "acc": task_score,
            "format_score": think_format_score,
            "think_format_score": think_format_score,
        }
    )
    return output


def _repo_root() -> Path:
    # .../training/verl/verl/utils/reward_score/rllm_code_reward.py -> repo root
    return Path(__file__).resolve().parents[5]


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


@lru_cache(maxsize=8)
def _official_lcb_modules(lm_style: str) -> tuple[dict[str, Any], Any]:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from evals.verifier.score_functions.code.livecodebench_official import (
        load_lcb_official_modules,
        resolve_lm_style,
    )

    modules = load_lcb_official_modules()
    return modules, resolve_lm_style(modules, lm_style)


def _compute_official_lcb_score(
    *,
    solution_str: str,
    extra_info: dict[str, Any],
    data_source: str,
    lm_style: str,
    timeout: int,
    worker_concurrency: int,
) -> dict[str, Any]:
    modules, style = _official_lcb_modules(lm_style)
    CodeGenerationProblem = modules["CodeGenerationProblem"]
    LanguageModelStore = modules["LanguageModelStore"]
    Scenario = modules["Scenario"]
    combine_results = modules["combine_results"]
    extract_instance_results = modules["extract_instance_results"]
    get_metrics = modules["get_metrics"]

    from types import SimpleNamespace

    metadata = _parse_metadata(extra_info.get("metadata"))
    problem = CodeGenerationProblem(**metadata)
    completions = [[str(solution_str or "")]]
    model = LanguageModelStore["Qwen/Qwen2.5-72B-Instruct"]
    combined_results = combine_results(Scenario.codegeneration, completions, model)
    eval_args = SimpleNamespace(
        scenario=Scenario.codegeneration,
        num_process_evaluate=max(1, int(worker_concurrency)),
        timeout=max(1, int(timeout)),
    )
    metrics = get_metrics(Scenario.codegeneration, eval_args, [problem], combined_results)
    graded = extract_instance_results(metrics[1])
    passed = bool(graded and graded[0] and graded[0][0])
    return _with_think_format_score({
        "score": float(passed),
        "acc": float(passed),
        "official_lcb_passed": float(passed),
        "dataset": str(extra_info.get("dataset") or data_source),
        "sample_id": str(extra_info.get("sample_id", "")),
        "request_id": str(extra_info.get("request_id", "")),
        "lm_style": style.value,
    }, solution_str)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    max_tests: int = 15,
    timeout: int = 6,
    testcase_max_workers: int = 1,
    scoring_mode: str = "rllm",
    lcb_lm_style: str = "CodeQwenInstruct",
    lcb_worker_concurrency: int = 24,
    **_: Any,
) -> float | dict[str, Any]:
    """Thin Verl adapter around rLLM's code reward implementation."""
    extra_info = extra_info or {}
    if str(scoring_mode or "").strip().lower() in {"official_lcb", "livecodebench_official", "eval_official"}:
        return _compute_official_lcb_score(
            solution_str=solution_str,
            extra_info=extra_info,
            data_source=data_source,
            lm_style=lcb_lm_style,
            timeout=timeout,
            worker_concurrency=lcb_worker_concurrency,
        )

    max_tests = _as_positive_int(max_tests, 15)
    timeout = _as_positive_int(timeout, 6)
    testcase_max_workers = _as_positive_int(testcase_max_workers, 1)
    reward_data_source = DATA_SOURCE_ALIASES.get(str(data_source), str(data_source))
    task_info = {
        "problem": None,
        "problem_type": RewardType.CODE,
        "data_source": reward_data_source,
        "ground_truth": _load_ground_truth(ground_truth),
    }
    reward_output = _reward_fn(max_tests, timeout, testcase_max_workers)(
        task_info=task_info,
        action=_ensure_code_block(solution_str or ""),
    )
    return _with_think_format_score(float(reward_output.reward), solution_str)

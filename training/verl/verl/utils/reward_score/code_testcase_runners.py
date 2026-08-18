"""Testcase normalization, code extraction, and per-testcase runner functions.

This module has two kinds of callables:

* **Driver-side helpers** (``normalize_test_cases``, ``decode_official_testcases``,
  ``extract_train_code``, ``extract_official_code``, ``is_official_lcb_metadata``)
  run in the trainer process while building a reward plan. They are pure-Python
  and import the heavy code-execution stack lazily.

* **Worker-side runners** (``run_rllm_lcb_testcases``, ``run_official_lcb_testcases``)
  are the ``fn`` handed to :class:`~verl.trainer.ppo.subprocess_pool.Job`. They run
  inside the throwaway subprocess of a pool worker and return a
  ``{"passed": bool, "error_code": int}`` dict for each testcase. Official LCB
  testcase jobs execute the prefix up to the requested testcase, preserving the
  official scorer's compile/state/early-failure semantics while still allowing
  testcase-level fanout.

The validation path is deliberately routed through the *official* LiveCodeBench
runner (cloned at runtime) so RL val scores line up with the eval/SFT pipeline.
The training path uses the rLLM vendored runner, which accepts the heterogeneous
testcase shapes DeepCoder ships (TACO / PrimeIntellect / LCB) and needs no
problem-level metadata.
"""

from __future__ import annotations

import base64
import json
import pickle
import re
import sys
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n.*?```", re.DOTALL)


def _repo_root() -> Path:
    # .../training/verl/verl/utils/reward_score/code_testcase_runners.py -> repo root
    return Path(__file__).resolve().parents[5]


# --------------------------------------------------------------------------- #
# Testcase normalization (driver-side)
# --------------------------------------------------------------------------- #
def parse_jsonish(raw: Any) -> Any:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}


def load_private_test_cases(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
            if isinstance(decoded, str):
                decoded = json.loads(decoded)
            return decoded if isinstance(decoded, list) else []
    return []


def _select_longest_input_tests(tests: list[dict[str, Any]], max_tests: int | None) -> list[dict[str, Any]]:
    if max_tests is None or max_tests <= 0 or len(tests) <= max_tests:
        return list(tests)
    return sorted(tests, key=lambda test: len(str(test.get("input", ""))), reverse=True)[:max_tests]


def normalize_test_cases(raw_tests: Any, *, max_tests: int | None) -> list[dict[str, Any]]:
    """Coerce the many training ground-truth shapes into a flat list of testcases."""
    raw_tests = parse_jsonish(raw_tests)
    tests: list[dict[str, Any]] = []
    if isinstance(raw_tests, list):
        tests = [dict(test) for test in raw_tests]
    elif isinstance(raw_tests, dict) and {"public_test_cases", "private_test_cases"} <= raw_tests.keys():
        tests = decode_official_testcases(raw_tests)
    elif isinstance(raw_tests, dict):
        inputs = raw_tests.get("inputs") or []
        outputs = raw_tests.get("outputs") or []
        fn_name = raw_tests.get("fn_name")
        tests = [
            {
                "input": inp,
                "output": out,
                **({"testtype": "functional", "metadata": {"func_name": fn_name}} if fn_name else {}),
            }
            for inp, out in zip(inputs, outputs, strict=False)
        ]
    return _select_longest_input_tests(tests, max_tests)


def decode_official_testcases(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Public + private testcases from a LiveCodeBench problem metadata dict."""
    public = parse_jsonish(metadata.get("public_test_cases")) or []
    if not isinstance(public, list):
        public = []
    private = load_private_test_cases(metadata.get("private_test_cases"))
    tests = [dict(test) for test in [*public, *private]]

    problem_metadata = parse_metadata(metadata.get("metadata"))
    fn_name = problem_metadata.get("func_name")
    if fn_name:
        for test in tests:
            test["testtype"] = "functional"
            test_metadata = parse_metadata(test.get("metadata"))
            test_metadata["func_name"] = fn_name
            test["metadata"] = test_metadata
    return tests


def is_official_lcb_metadata(metadata: Any) -> bool:
    """True when a row carries a full LiveCodeBench problem (eligible for official scoring)."""
    if not isinstance(metadata, dict) or not metadata:
        return False
    has_id = "question_id" in metadata or "question_title" in metadata
    has_tests = "public_test_cases" in metadata or "private_test_cases" in metadata
    return bool(has_id and has_tests)


# --------------------------------------------------------------------------- #
# Code extraction (driver-side)
# --------------------------------------------------------------------------- #
def _ensure_code_block(solution: str) -> str:
    if CODE_BLOCK_RE.search(solution or ""):
        return solution
    return f"```python\n{solution}\n```"


def extract_train_code(completion: str) -> str | None:
    """Generic markdown code-block extraction used for training rows."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", _ensure_code_block(completion or ""), re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


@lru_cache(maxsize=8)
def _official_extract_code_fn(lm_style: str):
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from evals.verifier.score_functions.code.livecodebench_official import (
        load_lcb_official_modules,
        resolve_lm_style,
    )

    modules = load_lcb_official_modules()
    style = resolve_lm_style(modules, lm_style)
    extract_code = modules["extract_code"]
    return lambda completion: str(extract_code(completion, style) or "")


def extract_official_code(completion: str, lm_style: str) -> str:
    """Official LiveCodeBench extractor (matches eval/SFT extraction)."""
    return _official_extract_code_fn(lm_style)(completion or "")


# --------------------------------------------------------------------------- #
# Per-testcase runners (worker-side). Each returns one dict per input testcase.
# --------------------------------------------------------------------------- #
def _single_testcase_sample(test_case: dict[str, Any]) -> dict[str, Any]:
    import verl.utils.reward_score.rllm_code_reward  # noqa: F401 - puts vendor on sys.path
    from rllm.rewards.code_reward import postprocess_lcb_sample

    return postprocess_lcb_sample([test_case])


def _multi_testcase_sample(test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    import verl.utils.reward_score.rllm_code_reward  # noqa: F401 - puts vendor on sys.path
    from rllm.rewards.code_reward import postprocess_lcb_sample

    return postprocess_lcb_sample(test_cases)


def _interpret(res: Any, metadata: Any) -> dict[str, Any]:
    passed = bool(isinstance(res, list) and res and res[0] is True)
    error_code = 0
    output = {"passed": passed, "error_code": error_code}
    if isinstance(metadata, dict):
        try:
            error_code = int(metadata.get("error_code", 0) or 0)
        except (TypeError, ValueError):
            error_code = 0
        output["error_code"] = error_code
        for key in ("error", "error_message", "output"):
            if metadata.get(key):
                output[key] = str(metadata[key])
    return output


def _official_value_passed(value: Any) -> bool:
    try:
        return bool(value > 0)
    except TypeError:
        return value is True


def _parse_official_failure_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, list) and metadata:
        metadata = metadata[0]
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {"error": metadata}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _interpret_official_problem(res: Any, metadata: Any, expected_tests: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    result_values = list(res) if isinstance(res, list) else []
    failure_metadata = _parse_official_failure_metadata(metadata)
    failure_consumed = False
    for idx in range(expected_tests):
        value = result_values[idx] if idx < len(result_values) else False
        passed = _official_value_passed(value)
        item: dict[str, Any] = {"passed": passed, "error_code": 0}
        if not passed and not failure_consumed:
            failure_consumed = True
            if failure_metadata:
                try:
                    item["error_code"] = int(failure_metadata.get("error_code", 0) or 0)
                except (TypeError, ValueError):
                    item["error_code"] = 0
                for key in ("error", "error_message", "output"):
                    if failure_metadata.get(key):
                        item[key] = str(failure_metadata[key])
            elif value in (-1, -2, -3, -4):
                item["error_code"] = int(value)
        outputs.append(item)
    return outputs


def _rllm_run_test():
    import verl.utils.reward_score.rllm_code_reward  # noqa: F401 - puts vendor on sys.path
    from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test

    return lcb_run_test


@lru_cache(maxsize=1)
def _official_run_test():
    """Official LiveCodeBench testcase runner from the cloned repo.

    Falls back to the rLLM vendored copy (same algorithm) if the official import
    path is unavailable, so validation never hard-fails on an import mismatch.
    """
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from evals.verifier.score_functions.code.livecodebench_official import load_lcb_official_modules

        run_test = load_lcb_official_modules().get("run_test")
        if run_test is not None:
            return run_test
    except (ImportError, KeyError):
        pass
    return _rllm_run_test()


@lru_cache(maxsize=1)
def _official_check_correctness():
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from evals.verifier.score_functions.code.livecodebench_official import load_lcb_official_modules

        load_lcb_official_modules()
        from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

        return check_correctness
    except ImportError:
        return None


def run_rllm_lcb_testcases(*, code: str, test_cases: list[dict[str, Any]], timeout: int) -> list[dict[str, Any]]:
    run_test = _rllm_run_test()
    results = []
    for test_case in test_cases:
        try:
            res, metadata = run_test(_single_testcase_sample(test_case), test=code, debug=False, timeout=timeout)
            results.append(_interpret(res, metadata))
        except BaseException as exc:  # noqa: BLE001
            results.append({"passed": False, "error_code": -4, "error": repr(exc)})
    return results


def run_official_lcb_testcases(
    *,
    code: str,
    test_cases: list[dict[str, Any]],
    timeout: int,
    target_test_idx: int | None = None,
) -> list[dict[str, Any]]:
    try:
        check_correctness = _official_check_correctness()
        if check_correctness is not None:
            res, metadata = check_correctness(_multi_testcase_sample(test_cases), code, timeout=timeout, debug=False)
        else:
            run_test = _official_run_test()
            res, metadata = run_test(_multi_testcase_sample(test_cases), test=code, debug=False, timeout=timeout)
        interpreted = _interpret_official_problem(res, metadata, len(test_cases))
        if target_test_idx is None:
            return interpreted
        if 0 <= target_test_idx < len(interpreted):
            return [interpreted[target_test_idx]]
        return [{"passed": False, "error_code": 0}]
    except BaseException as exc:  # noqa: BLE001
        if not test_cases:
            return []
        return [{"passed": False, "error_code": -4, "error": repr(exc)}] + [
            {"passed": False, "error_code": 0} for _ in range(len(test_cases) - 1)
        ]


# --------------------------------------------------------------------------- #
# Warmup hooks (run in the actor process so forked children inherit the imports)
# --------------------------------------------------------------------------- #
def warmup_rllm_runner() -> None:
    _rllm_run_test()
    from rllm.rewards.code_reward import postprocess_lcb_sample  # noqa: F401


def warmup_official_runner() -> None:
    _official_check_correctness()
    _official_run_test()
    import verl.utils.reward_score.rllm_code_reward  # noqa: F401 - puts vendor on sys.path
    from rllm.rewards.code_reward import postprocess_lcb_sample  # noqa: F401

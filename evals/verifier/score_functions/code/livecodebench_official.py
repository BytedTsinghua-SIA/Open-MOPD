from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.common.utils import parse_metadata, repo_path


def load_lcb_official_modules() -> dict[str, Any]:
    lcb_repo = repo_path("livecodebench")
    os.environ.setdefault(
        "LIVECODEBENCH_CODEGEN_LITE_DIR",
        str(lcb_repo.parents[2] / "benchmark_assets" / "livecodebench" / "code_generation_lite"),
    )
    if str(lcb_repo) not in sys.path:
        sys.path.insert(0, str(lcb_repo))
    if "anthropic" not in sys.modules:
        anthropic = types.ModuleType("anthropic")
        anthropic.HUMAN_PROMPT = "\n\nHuman:"
        anthropic.AI_PROMPT = "\n\nAssistant:"
        sys.modules["anthropic"] = anthropic
    cwd = Path.cwd()
    try:
        os.chdir(lcb_repo)
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # type: ignore
        from lcb_runner.evaluation import extract_instance_results  # type: ignore
        from lcb_runner.lm_styles import LanguageModelStore, LMStyle  # type: ignore
        from lcb_runner.runner.scenario_router import combine_results, get_metrics  # type: ignore
        from lcb_runner.utils.extraction_utils import extract_code  # type: ignore
        from lcb_runner.utils.scenarios import Scenario  # type: ignore

        try:
            # Testcase-level runner used for per-testcase parallel scoring. Kept
            # optional so eval paths that only need problem-level metrics still
            # load even if the upstream layout changes.
            from lcb_runner.evaluation.testing_util import run_test  # type: ignore
        except ImportError:
            run_test = None
    finally:
        os.chdir(cwd)

    return {
        "CodeGenerationProblem": CodeGenerationProblem,
        "extract_instance_results": extract_instance_results,
        "LanguageModelStore": LanguageModelStore,
        "LMStyle": LMStyle,
        "combine_results": combine_results,
        "get_metrics": get_metrics,
        "extract_code": extract_code,
        "Scenario": Scenario,
        "run_test": run_test,
    }


def resolve_lm_style(modules: dict[str, Any], lm_style: str) -> Any:
    LMStyle = modules["LMStyle"]
    return LMStyle[lm_style] if lm_style in LMStyle.__members__ else LMStyle(lm_style)


def prepare_lcb_groups(df: pd.DataFrame, lm_style: str) -> tuple[list[Any], list[list[str]], list[list[str]], list[Any]]:
    modules = load_lcb_official_modules()
    CodeGenerationProblem = modules["CodeGenerationProblem"]
    extract_code = modules["extract_code"]
    style = resolve_lm_style(modules, lm_style)

    problems = []
    completions = []
    generations = []
    outputs = []
    group_key = "sample_id" if "sample_id" in df.columns else "request_id"
    for _, problem_df in df.groupby(group_key, sort=False):
        first_row = problem_df.iloc[0]
        problem = CodeGenerationProblem(**parse_metadata(first_row))
        problem_completions = [str(value or "") for value in problem_df["completion"].tolist()]
        problem_generations = [extract_code(completion, style) for completion in problem_completions]
        problems.append(problem)
        completions.append(problem_completions)
        generations.append(problem_generations)
        outputs.append(problem.insert_output(problem_completions, problem_generations))
    return problems, completions, generations, outputs


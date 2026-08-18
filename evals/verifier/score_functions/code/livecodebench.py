from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.code.livecodebench_official import (
    load_lcb_official_modules,
    prepare_lcb_groups,
    resolve_lm_style,
)
from evals.verifier.score_functions.code.livecodebench_parallel import (
    score_lcb_multi_process,
    score_lcb_single_process,
)
from evals.verifier.score_functions.code.livecodebench_subsets import filter_lcb_subsets
from evals.verifier.score_functions.common.utils import log


def lcb_generation_passed(result: Any) -> bool:
    """LiveCodeBench marks each test with positive values on success and negative error codes on failure."""
    try:
        return all(value > 0 for value in result)
    except TypeError:
        return bool(result) is True


def score_lcb(
    df: pd.DataFrame,
    details_dir: Path | None,
    workers: int,
    progress_interval: int = 500,
    lm_style: str = "CodeQwenInstruct",
    *,
    data_dir: Path | None = None,
    subsets: str | None = None,
    process_workers: int = 1,
    worker_concurrency: int | None = None,
    timeout: int = 6,
) -> dict[str, Any]:
    try:
        modules = load_lcb_official_modules()
        style = resolve_lm_style(modules, lm_style)
    except Exception as exc:
        return {"error": f"dependency_missing_or_invalid_livecodebench: {exc}"}

    filter_extra: dict[str, Any] = {}
    if subsets:
        df, filter_extra = filter_lcb_subsets(df, data_dir, subsets)

    details = details_dir or Path("/tmp/openopd_lcb_score")
    details.mkdir(parents=True, exist_ok=True)
    log(f"livecodebench: preparing {len(df)} rollout rows with LMStyle={style.value}")
    problems, completions, generations, outputs = prepare_lcb_groups(df, lm_style)
    total_completions = sum(len(items) for items in generations)
    log(f"livecodebench: grouped into {len(problems)} problems and {total_completions} completions")

    output_stem = details.name if details.name else "livecodebench"
    outputs_path = details / f"{output_stem}_outputs.json"
    outputs_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"livecodebench: wrote outputs to {outputs_path}")

    process_workers = max(1, int(process_workers))
    if worker_concurrency is None:
        worker_concurrency = max(1, int(workers))
    worker_concurrency = max(1, int(worker_concurrency))
    if process_workers > 1 and len(problems) > 1:
        all_graded, all_metadata, evaluated = score_lcb_multi_process(
            problems,
            completions,
            generations,
            min(process_workers, len(problems)),
            worker_concurrency,
            timeout,
        )
    else:
        all_graded, all_metadata, evaluated = score_lcb_single_process(
            problems,
            completions,
            generations,
            max(1, int(workers)),
            progress_interval,
            timeout,
        )

    eval_path = details / f"{output_stem}_eval.json"
    eval_path.write_text(json.dumps(evaluated, ensure_ascii=False, indent=2), encoding="utf-8")
    pass_at_1 = (
        sum(sum(1.0 if bool(item) else 0.0 for item in graded_list) / len(graded_list) for graded_list in all_graded if graded_list)
        / len(all_graded)
        if all_graded
        else 0.0
    )
    return {
        "correct": pass_at_1,
        "scored_rows": len(problems),
        "metric": "livecodebench_pass_at_1",
        "official_extra": {
            "metrics": {"pass@1": pass_at_1},
            "outputs_path": str(outputs_path),
            "eval_path": str(eval_path),
            "metadata_count": len(all_metadata),
            "workers": max(1, int(workers)),
            "process_workers": process_workers,
            "worker_concurrency": worker_concurrency,
            "rollout_rows": len(df),
            "generations_per_problem": total_completions / len(problems) if problems else None,
            "lm_style": style.value,
            **filter_extra,
        },
    }


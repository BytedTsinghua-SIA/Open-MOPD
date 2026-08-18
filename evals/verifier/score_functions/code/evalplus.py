from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.common.utils import parse_metadata, repo_path


def score_evalplus(df: pd.DataFrame, dataset: str, details_dir: Path | None, workers: int) -> dict[str, Any]:
    try:
        if "termcolor" not in sys.modules:
            termcolor = types.ModuleType("termcolor")
            termcolor.colored = lambda text, *args, **kwargs: text
            termcolor.cprint = lambda text, *args, **kwargs: print(text)
            sys.modules["termcolor"] = termcolor
        if "evalplus.codegen" not in sys.modules:
            codegen = types.ModuleType("evalplus.codegen")

            def _run_codegen_stub(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("run_codegen is unavailable in verifier scoring")

            codegen.run_codegen = _run_codegen_stub
            sys.modules["evalplus.codegen"] = codegen
        evalplus_repo = repo_path("evalplus")
        if str(evalplus_repo) not in sys.path:
            sys.path.insert(0, str(evalplus_repo))
        import evalplus.evaluate as evalplus_evaluate  # type: ignore
        from evals.verifier.score_functions.code.evalplus_extract import extract_solution_code
    except Exception as exc:
        return {"error": f"dependency_missing: evalplus ({exc})"}

    details = details_dir or Path("/tmp/openopd_evalplus_score")
    details.mkdir(parents=True, exist_ok=True)
    samples_path = details / f"{dataset}_samples.jsonl"
    task_ids = set()
    with samples_path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            meta = parse_metadata(row)
            task_id = str(meta.get("task_id", row.get("sample_id")))
            if dataset == "mbpp_plus" and task_id.isdigit():
                task_id = f"Mbpp/{task_id}"
            entry_point = meta.get("entry_point", "")
            solution = extract_solution_code(str(row.get("completion", "")), entry_point=entry_point)
            handle.write(json.dumps({"task_id": task_id, "solution": solution}, ensure_ascii=False) + "\n")
            task_ids.add(task_id)
    eval_dataset = "humaneval" if dataset == "humaneval_plus" else "mbpp"
    parallel = max(1, int(workers))
    result_path = samples_path.with_suffix(".eval_results.json")
    if result_path.exists():
        result_path.unlink()
    evalplus_evaluate.evaluate(
        dataset=eval_dataset,
        samples=str(samples_path),
        base_only=False,
        parallel=parallel,
        i_just_wanna_run=True,
    )
    pass_at_k = {}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        pass_at_k = result.get("pass_at_k", {})
    plus_pass_at_1 = pass_at_k.get("plus", {}).get("pass@1")
    return {
        "correct": plus_pass_at_1,
        "scored_rows": len(task_ids),
        "metric": "evalplus_pass_at_1",
        "official_extra": {
            "samples_path": str(samples_path),
            "result_path": str(result_path),
            "pass_at_k": pass_at_k,
            "parallel": parallel,
            "rollout_rows": len(df),
        },
    }

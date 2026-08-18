from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.code.evalplus import score_evalplus
from evals.verifier.score_functions.code.livecodebench import lcb_generation_passed, score_lcb
from evals.verifier.score_functions.common.behavior import compute_behavior_metrics, is_repetitive_completion
from evals.verifier.score_functions.common.utils import DEFAULT_DATA_DIR, build_unscored, log, official_total
from evals.verifier.score_functions.instruction_following.common import (
    filter_kwargs_for_instructions,
    normalize_kwargs,
    summarize_instruction_outputs,
)
from evals.verifier.score_functions.instruction_following.ifbench import score_ifbench
from evals.verifier.score_functions.instruction_following.ifeval import score_ifeval
from evals.verifier.score_functions.math.aime import extract_aime_final_answer, score_aime, score_aime_row


MATH_DATASETS = {"aime24", "aime25"}
IFEVAL_DATASETS = {"ifeval", "ifbench_mt_ifeval"}
IFBENCH_DATASETS = {"ifbench_test", "ifbench_mt_ifbench"}
EVALPLUS_DATASETS = {"humaneval_plus", "mbpp_plus"}
LCB_DATASETS = {"livecodebench_v5", "livecodebench_v6", "livecodebench_full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OpenOPD eval rollout parquet files.")
    parser.add_argument("--rollout", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Full parquet directory for official denominators.")
    parser.add_argument("--details-dir", type=Path, help="Optional directory for official evaluator intermediate files.")
    parser.add_argument("--workers", type=int, default=160, help="Parallel evaluator workers for code benchmarks.")
    parser.add_argument("--read-workers", type=int, default=8, help="Parallel workers for reading rollout parquet files.")
    parser.add_argument("--lcb-progress-interval", type=int, default=500, help="Log after each LiveCodeBench batch of this many completions.")
    parser.add_argument("--lcb-lm-style", default="CodeQwenInstruct", help="LiveCodeBench LMStyle used for official code extraction.")
    parser.add_argument("--lcb-process-workers", type=int, default=160, help="Outer LiveCodeBench scoring process workers.")
    parser.add_argument(
        "--lcb-worker-concurrency",
        type=int,
        default=24,
        help="LiveCodeBench evaluator concurrency inside each process worker.",
    )
    parser.add_argument("--lcb-timeout", type=int, default=6, help="LiveCodeBench per-test timeout.")
    parser.add_argument(
        "--lcb-subsets",
        help="Optional comma-separated subset filter, e.g. v5,v6 for a livecodebench_full rollout.",
    )
    return parser.parse_args()


def score_dataset(
    df: pd.DataFrame,
    data_dir: Path,
    details_root: Path | None,
    workers: int = 160,
    lcb_progress_interval: int = 500,
    lcb_lm_style: str = "CodeQwenInstruct",
    lcb_process_workers: int = 160,
    lcb_worker_concurrency: int | None = 24,
    lcb_timeout: int = 6,
    lcb_subsets: str | None = None,
) -> dict[str, Any]:
    dataset = str(df["dataset"].iloc[0]) if len(df) and "dataset" in df else "unknown"
    total = official_total(dataset, data_dir)
    details_dir = details_root / dataset if details_root else None
    behavior = compute_behavior_metrics(df)

    if dataset in MATH_DATASETS:
        result = score_aime(df, dataset)
        total = result["total"]
    elif dataset in IFEVAL_DATASETS:
        result = score_ifeval(df, details_dir)
    elif dataset in IFBENCH_DATASETS:
        result = score_ifbench(df, details_dir)
    elif dataset in EVALPLUS_DATASETS:
        result = score_evalplus(df, dataset, details_dir, workers)
    elif dataset in LCB_DATASETS:
        result = score_lcb(
            df,
            details_dir,
            workers,
            lcb_progress_interval,
            lcb_lm_style,
            data_dir=data_dir,
            subsets=lcb_subsets,
            process_workers=lcb_process_workers,
            worker_concurrency=lcb_worker_concurrency,
            timeout=lcb_timeout,
        )
    else:
        return build_unscored(dataset, total, "unknown_dataset")

    if "error" in result:
        return build_unscored(dataset, total, str(result["error"]))

    correct = result["correct"]
    scored_rows = result["scored_rows"]
    if dataset in LCB_DATASETS:
        total = scored_rows
    metric = result["metric"]
    extra = result.get("official_extra", {})

    return {
        "dataset": dataset,
        "metric": metric,
        "scored_rows": scored_rows,
        "correct": correct,
        "official_total": total,
        "official": correct / total if isinstance(correct, int) and total else (correct if isinstance(correct, float) else None),
        "live": correct / scored_rows if isinstance(correct, int) and scored_rows else (correct if isinstance(correct, float) else None),
        "official_extra": extra,
        **behavior,
    }


def is_remote_path(path: str) -> bool:
    return "://" in path


def read_rollout(path: str | Path) -> tuple[str | Path, pd.DataFrame, float]:
    started = time.monotonic()
    path_text = str(path)
    read_path: str | Path = path_text if is_remote_path(path_text) else Path(path_text)
    size_mb = None
    if isinstance(read_path, Path) and read_path.exists():
        size_mb = read_path.stat().st_size / (1024 * 1024)
    size_text = f", {size_mb:.1f} MiB" if size_mb is not None else ""
    log(f"read start: {path_text}{size_text}")
    df = pd.read_parquet(read_path)
    elapsed = time.monotonic() - started
    log(f"read done: {path_text} rows={len(df)} elapsed={elapsed:.1f}s")
    return path, df, elapsed


def main() -> None:
    args = parse_args()
    frames = []
    rollout_paths = list(args.rollout)
    read_workers = max(1, int(args.read_workers))
    log(f"loading {len(rollout_paths)} rollout parquet files with {read_workers} read workers")
    if read_workers == 1 or len(rollout_paths) <= 1:
        for index, path in enumerate(rollout_paths, start=1):
            _, df, _elapsed = read_rollout(path)
            if not df.empty:
                frames.append(df)
            log(f"loaded files: {index}/{len(rollout_paths)}")
    else:
        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            futures = [executor.submit(read_rollout, path) for path in rollout_paths]
            for index, future in enumerate(as_completed(futures), start=1):
                _path, df, _elapsed = future.result()
                if not df.empty:
                    frames.append(df)
                log(f"loaded files: {index}/{len(rollout_paths)}")
    if frames:
        log(f"concatenating {len(frames)} non-empty dataframes")
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()
    log(f"combined rollout rows={len(combined)}")
    details = []
    if not combined.empty:
        for _, dataset_df in combined.groupby("dataset", sort=False):
            dataset = str(dataset_df["dataset"].iloc[0])
            log(f"dataset start: {dataset} rows={len(dataset_df)}")
            started = time.monotonic()
            details.append(
                score_dataset(
                    dataset_df,
                    args.data_dir,
                    args.details_dir,
                    args.workers,
                    args.lcb_progress_interval,
                    args.lcb_lm_style,
                    args.lcb_process_workers,
                    args.lcb_worker_concurrency,
                    args.lcb_timeout,
                    args.lcb_subsets,
                )
            )
            log(f"dataset done: {dataset} elapsed={time.monotonic() - started:.1f}s")
    payload = {"results": details, "macro_official": None, "macro_live": None}
    official_scores = [row["official"] for row in details if row.get("official") is not None]
    live_scores = [row["live"] for row in details if row.get("live") is not None]
    if official_scores:
        payload["macro_official"] = sum(float(x) for x in official_scores) / len(official_scores)
    if live_scores:
        payload["macro_live"] = sum(float(x) for x in live_scores) / len(live_scores)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")
    display = pd.DataFrame(details)
    if "official_extra" in display.columns:
        display = display.drop(columns=["official_extra"])
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

"""Score eval rollout parquets IN-PLACE on the eval node (cluster CPUs).

This is the scoring half of the scored-eval task type: the rollout engine dumps
unscored rollouts; instead of pulling them to a low-CPU host to grade, we grade right
here on the eval node (hundreds of CPUs). Two outputs:
  - a per-row ``score`` column written BACK INTO each rollout parquet (in-place), so the
    rollout result carries its own judgement (1.0/0.0 pass per completion);
  - a ``scores.json`` summary (avg@k / accuracy) next to the rollouts.

Dispatches by dataset to the SAME official scorers as offline grading + RL val (aligned):
  math (aime24/25) -> boxed-match vs reward_model.ground_truth
  code (livecodebench_v5/v6) -> official LCB (decode_official_testcases +
      extract_official_code(CodeQwenInstruct) + run_official_lcb_testcases), per-row pass
  IF (ifeval / ifbench_*) -> evals.verifier instruction_following (summary; per-row NaN)

Usage:
    python -m evals.score_rollouts --rollout-dir <dir> --dataset <name> --out <scores.json>
        [--workers N] [--lm-style CodeQwenInstruct] [--timeout 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training" / "verl"))


def _load_with_src(rollout_dir: str):
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.parquet")))
    files = [f for f in files if not f.endswith("_scored.parquet")]
    if not files:
        raise FileNotFoundError(f"no parquet under {rollout_dir}")
    frames = []
    for f in files:
        d = pq.read_table(f).to_pandas()
        d["_src_file"] = f
        frames.append(d)
    return files, pd.concat(frames, ignore_index=True)


# ---- per-row scorers -------------------------------------------------------
def _last_boxed(s):
    s = str(s); i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1:k]
    return None


def _norm(x):
    if x is None:
        return None
    x = str(x).strip().replace(",", "").replace(" ", "").replace("\\!", "").rstrip(".")
    m = re.fullmatch(r"-?\d+", x)
    return m.group(0) if m else x


def _math_rows(df):
    # Use the repository's canonical AIME row scorer instead of duplicating a
    # stricter string matcher here.  AIME references are conventionally stored
    # as zero-padded three-digit strings (for example ``"033"``), while models
    # naturally emit ``\boxed{33}``.  The old local matcher treated those as
    # different and produced large false-negative formal-eval regressions.
    from evals.verifier.score_functions.math.aime import score_aime_row

    return [float(score_aime_row(row)) for _, row in df.iterrows()]


def _grade_lcb_row(args):
    comp, md_raw, lm_style, timeout = args
    try:
        from verl.utils.reward_score.code_testcase_runners import (
            decode_official_testcases, is_official_lcb_metadata,
            extract_official_code, run_official_lcb_testcases,
        )
        md = json.loads(md_raw) if isinstance(md_raw, str) else md_raw
        if not is_official_lcb_metadata(md):
            return float("nan")
        tests = decode_official_testcases(md)
        code = extract_official_code(str(comp), lm_style)
        if not code or not tests:
            return 0.0
        res = run_official_lcb_testcases(code=code, test_cases=tests, timeout=timeout)
        return 1.0 if (res and all(r.get("passed") for r in res)) else 0.0
    except Exception:
        return 0.0


def _code_rows(df, workers, lm_style, timeout):
    from concurrent.futures import ProcessPoolExecutor
    args = [(c, m, lm_style, timeout) for c, m in zip(df["completion"].astype(str), df["metadata"])]
    print(f"[score_rollouts] grading {len(args)} code completions with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_grade_lcb_row, args, chunksize=4))


def _grade_code_testcases_row(args):
    """Grade one code-farming completion against its row's reward_model.ground_truth
    test cases (taco/primeintellect/codecontests/livecodebench all ship stdin/stdout
    lists ``[{"input","output",...}]``). Reuses the official LCB stdin runner — no pyext."""
    comp, gt_raw, lm_style, timeout = args
    try:
        from verl.utils.reward_score.code_testcase_runners import (
            extract_official_code, run_official_lcb_testcases,
        )
        tests = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        # normalize formats: raw source ships {"inputs":[...],"outputs":[...]} (dict of
        # arrays); filtered/official ships [{"input","output",...}] (list of dicts).
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            tests = [{"input": i, "output": o}
                     for i, o in zip(tests["inputs"], tests["outputs"])]
        if not tests:
            return float("nan")
        code = extract_official_code(str(comp), lm_style)
        if not code:
            return 0.0
        res = run_official_lcb_testcases(code=code, test_cases=list(tests), timeout=timeout)
        return 1.0 if (res and all(r.get("passed") for r in res)) else 0.0
    except Exception:
        return 0.0


def _farming_gt(md_raw):
    """Pull the testcase ground_truth out of a farming row's flat ``metadata`` JSON
    (input is pre-staged FLAT — prompt + metadata string — so the eval input-prep doesn't
    choke on nested struct columns). metadata = {"data_source","ground_truth","index"}."""
    md = json.loads(md_raw) if isinstance(md_raw, str) else (md_raw or {})
    if not isinstance(md, dict):
        return None
    gt = md.get("ground_truth")
    if gt is None and isinstance(md.get("reward_model"), dict):
        gt = md["reward_model"].get("ground_truth")
    return gt


def _code_farming_rows(df, workers, lm_style, timeout):
    """Code data-farming: grade each completion against the per-row testcases packed into
    the flat ``metadata`` column (data_source/ground_truth/index)."""
    from concurrent.futures import ProcessPoolExecutor
    args = [(c, _farming_gt(m), lm_style, timeout)
            for c, m in zip(df["completion"].astype(str), df["metadata"])]
    print(f"[score_rollouts] grading {len(args)} code-farming completions with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_grade_code_testcases_row, args, chunksize=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    # default to (almost) ALL CPUs the container actually has — LCB grading is CPU-bound
    # (each completion runs the official test harness in a subprocess); sched_getaffinity
    # is the real cgroup-available count (more accurate than os.cpu_count in a container).
    try:
        _ncpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        _ncpu = os.cpu_count() or 16
    ap.add_argument("--workers", type=int, default=max(8, _ncpu - 2))
    ap.add_argument("--lm-style", default="CodeQwenInstruct")
    ap.add_argument("--timeout", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=30000,
                    help="generation cap; a completion is 'truncated' if completion_tokens >= this")
    a = ap.parse_args()

    files, df = _load_with_src(a.rollout_dir)
    ds = a.dataset.lower()
    per_row = None
    if ds.startswith("aime"):
        per_row = _math_rows(df)
    elif "livecodebench" in ds or ds.startswith("lcb"):
        per_row = _code_rows(df, a.workers, a.lm_style, a.timeout)
    elif "deepscaler" in ds or "deepcoder" in ds or "farm" in ds or ("code" in ds and "livecodebench" not in ds):
        # code data-farming: testcases packed into the flat metadata column
        per_row = _code_farming_rows(df, a.workers, a.lm_style, a.timeout)
    elif ds == "ifeval" or "ifbench" in ds or ds.startswith("if"):
        # IF: strict prompt-level pass, now exposed PER ROW so it lands in the parquet too.
        from evals.verifier.score_functions.instruction_following.ifeval import score_ifeval
        from evals.verifier.score_functions.instruction_following.ifbench import score_ifbench
        fn = score_ifeval if ds == "ifeval" else score_ifbench
        r = fn(df.drop(columns=["_src_file"]), None)
        per_row = r.get("per_row_strict")  # aligned to df row order
        if per_row is None or len(per_row) != len(df):
            # fallback: summary-only if per-row not available / misaligned
            ex = r.get("official_extra", {})
            summary = {"metric": "strict_prompt", "correct": r["correct"], "scored_rows": r["scored_rows"],
                       "pct": 100.0 * ex.get("strict", {}).get("prompt_level_accuracy", r["correct"] / max(1, r["scored_rows"]))}
            out = {"dataset": a.dataset, "rows": len(df), **summary}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=2))
            print(f"[score_rollouts] {a.dataset}: {summary['pct']:.2f}% ({summary['correct']}/{summary['scored_rows']}) -> {a.out} (summary-only)")
            return
    else:
        print(f"[score_rollouts] no scorer for dataset {a.dataset!r}; leaving rollouts unscored.")
        return

    df["score"] = per_row
    graded = [x for x in per_row if x == x]  # drop NaN
    npass = sum(1 for x in graded if x); n = len(graded)
    summary = {"metric": "avg_at_k_rows", "correct": npass, "scored_rows": n,
               "total": len(df), "pct": 100.0 * npass / max(1, n)}
    # write the per-row score BACK INTO each source parquet (in-place), then it gets uploaded
    for f in files:
        sub = df[df["_src_file"] == f].drop(columns=["_src_file"])
        sub.to_parquet(f, index=False)

    out = {"dataset": a.dataset, "rows": len(df), **summary}
    # truncation rate: completion hit the generation cap (completion_tokens >= max_tokens)
    if "completion_tokens" in df.columns:
        ct = df["completion_tokens"].astype(float)
        ntrunc = int((ct >= a.max_tokens).sum())
        out["truncated"] = ntrunc
        out["truncation_rate"] = 100.0 * ntrunc / max(1, len(df))
        out["max_tokens"] = a.max_tokens
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    tr = f" trunc={out.get('truncation_rate', float('nan')):.1f}%" if "truncation_rate" in out else ""
    print(f"[score_rollouts] {a.dataset}: {summary['pct']:.2f}% ({summary['correct']}/{summary['scored_rows']}){tr} "
          f"-> {a.out} (+per-row score column written back)")


if __name__ == "__main__":
    main()

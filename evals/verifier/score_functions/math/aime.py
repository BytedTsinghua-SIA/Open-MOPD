from __future__ import annotations

import re

import pandas as pd


def extract_aime_final_answer(text: str) -> str:
    stripped = str(text or "").strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", stripped)
    if boxed:
        digits = re.findall(r"-?\d+", boxed[-1].replace(",", ""))
        return digits[-1] if digits else boxed[-1].strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"\d{1,3}", line):
            return line
        digits = re.findall(r"-?\d+", line.replace(",", ""))
        if len(digits) == 1 and re.fullmatch(r"\d{1,3}", digits[0]):
            return digits[0]
    matches = re.findall(r"-?\d+", stripped.replace(",", ""))
    return matches[-1] if matches else stripped


def score_aime_row(row: pd.Series) -> bool:
    pred = extract_aime_final_answer(str(row.get("completion", "")))
    ref = extract_aime_final_answer(str(row.get("answer", "")))
    try:
        return int(pred) == int(ref)
    except (TypeError, ValueError):
        return pred == ref


def score_aime(df: pd.DataFrame, dataset: str) -> dict:
    correct = int(df.apply(score_aime_row, axis=1).sum())
    scored_rows = int(len(df))
    return {
        "correct": correct,
        "scored_rows": scored_rows,
        "total": scored_rows,
        "metric": "avg_at_k_rows",
        "official_extra": {"unique_samples": int(df["sample_id"].nunique()) if "sample_id" in df.columns else None},
    }

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


def repeated_ngram_ratio(text: str, n: int = 6) -> float:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(ngrams), 1)


def trailing_line_repeat_ratio(text: str, tail_lines: int = 40) -> float:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0
    tail = lines[-tail_lines:]
    counts = Counter(tail)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(tail), 1)


def repeated_phrase_score(text: str) -> int:
    phrases = [
        "let me denote",
        "wait,",
        "let me think",
        "let me write",
        "first, let me",
        "okay, let me",
    ]
    lowered = str(text or "").lower()
    return max((lowered.count(phrase) for phrase in phrases), default=0)


def is_repetitive_completion(text: str) -> bool:
    if not text:
        return False
    if repeated_phrase_score(text) >= 8:
        return True
    if repeated_ngram_ratio(text) >= 0.18:
        return True
    if trailing_line_repeat_ratio(text) >= 0.25:
        return True
    return False


def compute_behavior_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    rows = int(len(df))
    if rows == 0:
        return {
            "trunc_rows": 0,
            "repeat_rows": 0,
            "trunc_repeat_rows": 0,
            "trunc_rate": None,
            "repeat_rate": None,
            "trunc_repeat_rate": None,
        }

    if {"completion_tokens", "max_tokens"}.issubset(df.columns):
        trunc_mask = df["completion_tokens"] >= df["max_tokens"]
    else:
        trunc_mask = pd.Series([False] * rows, index=df.index)

    if "completion" in df.columns:
        repeat_mask = df["completion"].fillna("").map(is_repetitive_completion)
    else:
        repeat_mask = pd.Series([False] * rows, index=df.index)

    trunc_rows = int(trunc_mask.sum())
    repeat_rows = int(repeat_mask.sum())
    trunc_repeat_rows = int((trunc_mask & repeat_mask).sum())
    return {
        "trunc_rows": trunc_rows,
        "repeat_rows": repeat_rows,
        "trunc_repeat_rows": trunc_repeat_rows,
        "trunc_rate": trunc_rows / rows,
        "repeat_rate": repeat_rows / rows,
        "trunc_repeat_rate": trunc_repeat_rows / rows,
    }

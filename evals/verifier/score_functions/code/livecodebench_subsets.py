from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evals.verifier.score_functions.common.utils import parse_metadata


def metadata_question_id(metadata: dict[str, Any]) -> str | None:
    for key in ("question_id", "questionId", "id"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return None


def load_lcb_question_ids(path: Path) -> set[str]:
    try:
        frame = pd.read_parquet(path, columns=["metadata"])
    except Exception:
        frame = pd.read_parquet(path)
    ids = set()
    for metadata in frame.get("metadata", []):
        if isinstance(metadata, str) and metadata.strip():
            metadata = json.loads(metadata)
        if isinstance(metadata, dict):
            question_id = metadata_question_id(metadata)
            if question_id is not None:
                ids.add(question_id)
    return ids


def filter_lcb_subsets(df: pd.DataFrame, data_dir: Path | None, subsets: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not subsets:
        return df, {}
    requested = [item.strip() for item in subsets.split(",") if item.strip()]
    if not requested:
        return df, {}
    if data_dir is None:
        return df, {"requested_subsets": requested, "filter_note": "missing_data_dir"}

    wanted_ids: set[str] = set()
    sources = []
    for subset in requested:
        dataset = subset
        if subset.lower() in {"v5", "5"}:
            dataset = "livecodebench_v5"
        elif subset.lower() in {"v6", "6"}:
            dataset = "livecodebench_v6"
        path = data_dir / f"{dataset}.parquet"
        if not path.exists():
            sources.append({"subset": subset, "path": str(path), "exists": False, "ids": 0})
            continue
        ids = load_lcb_question_ids(path)
        wanted_ids.update(ids)
        sources.append({"subset": subset, "path": str(path), "exists": True, "ids": len(ids)})
    if not wanted_ids:
        return df, {"requested_subsets": requested, "filter_note": "no_subset_ids_found", "sources": sources}

    keep_mask = []
    for _, row in df.iterrows():
        question_id = metadata_question_id(parse_metadata(row))
        keep_mask.append(question_id in wanted_ids)
    filtered = df.loc[keep_mask].copy()
    return filtered, {
        "requested_subsets": requested,
        "sources": sources,
        "input_rows_before_subset_filter": int(len(df)),
        "rows_after_subset_filter": int(len(filtered)),
        "subset_question_ids": len(wanted_ids),
    }


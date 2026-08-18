from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


VERIFIER_DIR = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = VERIFIER_DIR / "third_party"
DEFAULT_DATA_DIR = VERIFIER_DIR.parent / "data" / "parquet"
START_TIME = time.monotonic()

DATASET_TOTALS = {
    "aime24": 30 * 32,
    "aime25": 30 * 32,
    "humaneval_plus": 164 * 10,
    "mbpp_plus": 378 * 10,
    "livecodebench_v5": 167,
    "livecodebench_v6": 175,
    "livecodebench_full": 1055,
    "ifeval": 541,
    "ifbench_test": 300,
    "ifbench_mt_ifbench": 1387,
    "ifbench_mt_ifeval": 1774,
}


def log(message: str) -> None:
    elapsed = time.monotonic() - START_TIME
    print(f"[score] +{elapsed:.1f}s {message}", file=sys.stderr, flush=True)


def load_repo_lock() -> dict:
    candidates = [
        THIRD_PARTY_ROOT / "repos.lock.json",
        VERIFIER_DIR.parent.parent / "third_party" / "repos.lock.json",
        VERIFIER_DIR.parent.parent.parent / "ModelMerging" / "third_party" / "repos.lock.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"repos.lock.json not found in: {candidates}")


def repo_path(name: str) -> Path:
    env_key = f"OPENOPD_{name.upper()}_REPO"
    if os.environ.get(env_key):
        return Path(os.environ[env_key]).expanduser()
    entry = load_repo_lock()["repos"][name]
    rel = Path(entry.get("checkout", ""))
    if rel.parts[:2] == ("third_party", "repos"):
        rel = Path(*rel.parts[2:])
    candidates = [
        THIRD_PARTY_ROOT / "repos" / rel,
        VERIFIER_DIR.parent.parent / "third_party" / "repos" / rel,
        VERIFIER_DIR.parent.parent.parent / "ModelMerging" / "third_party" / "repos" / rel,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def official_total(dataset: str, data_dir: Path) -> int | None:
    path = data_dir / f"{dataset}.parquet"
    if path.exists():
        return len(pd.read_parquet(path, columns=["request_id"]))
    return DATASET_TOTALS.get(dataset)


def parse_metadata(row: pd.Series) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata.strip():
        return json.loads(metadata)
    return {}


def build_unscored(dataset: str, total: int | None, note: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scored_rows": 0,
        "correct": None,
        "official_total": total,
        "official": None,
        "live": None,
        "note": note,
    }

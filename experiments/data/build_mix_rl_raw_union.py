# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build a deterministic MixRL pool from raw domains.

Unlike ``build_mix_rl_union.py``, this builder does not restore a selection
made by an earlier data-farming run.  It samples directly from the original
Math, Code, and IF RL parquets and emits batches with an exact 1:1:1 domain
ratio.  Code rows tagged ``lcbv5`` are excluded by default to avoid training
on the validation family; this is source decontamination, not reward-based
difficulty filtering.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


DOMAINS = ("math", "code", "if")
PROMPT_TYPE = pa.list_(
    pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
)
REWARD_MODEL_TYPE = pa.struct(
    [pa.field("style", pa.string()), pa.field("ground_truth", pa.string())]
)
EXTRA_INFO_TYPE = pa.struct(
    [
        pa.field("split", pa.string()),
        pa.field("index", pa.string()),
        pa.field("source_config", pa.string()),
        pa.field("sample_id", pa.int64()),
        pa.field("raw_prompt", pa.string()),
        pa.field("instruction_id_list", pa.list_(pa.string())),
        pa.field("instruction_kwargs_json", pa.list_(pa.string())),
        pa.field("dataset", pa.string()),
        pa.field("agent_name", pa.string()),
        pa.field("family", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("source_index", pa.string()),
        pa.field("mix_index", pa.int64()),
    ]
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("prompt", PROMPT_TYPE),
        pa.field("ability", pa.string()),
        pa.field("reward_model", REWARD_MODEL_TYPE),
        pa.field("extra_info", EXTRA_INFO_TYPE),
        pa.field("domain", pa.string()),
    ]
)


def _priority(seed: int, domain: str, source_position: int) -> int:
    payload = f"{seed}:{domain}:{source_position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _iter_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    # DeepCoder's nested testcase column spans multiple physical parquet
    # chunks.  PyArrow cannot convert a few of those row groups directly
    # (``Nested data conversions not implemented for chunked array outputs``),
    # while Polars' native parquet reader handles the file losslessly.  This is
    # also the reader used by the cluster runtime for the same source.
    frame = pl.read_parquet(path, use_pyarrow=False, low_memory=True, rechunk=False)
    yield from enumerate(frame.iter_rows(named=True))


def _normalize_record(
    row: dict[str, Any], *, domain: str, source_position: int, mix_index: int
) -> dict[str, Any]:
    extra = dict(row.get("extra_info") or {})
    normalized_extra = {field.name: None for field in EXTRA_INFO_TYPE}
    for key in normalized_extra:
        if key in extra:
            normalized_extra[key] = extra[key]
    normalized_extra.update(
        {
            "index": f"{domain}-{mix_index}",
            "domain": domain,
            "source_index": str(extra.get("index", source_position)),
            "mix_index": mix_index,
        }
    )
    if normalized_extra["sample_id"] is not None:
        normalized_extra["sample_id"] = int(normalized_extra["sample_id"])
    return {
        "data_source": str(row["data_source"]),
        "prompt": row["prompt"],
        "ability": str(row.get("ability") or domain),
        "reward_model": row["reward_model"],
        "extra_info": normalized_extra,
        "domain": domain,
    }


def _select_domain(
    path: Path,
    *,
    domain: str,
    rows: int,
    seed: int,
    excluded_data_sources: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Keep the rows with the smallest deterministic hash priorities.  The heap
    # stores negative priorities so its root is the worst retained item.
    heap: list[tuple[int, int, dict[str, Any]]] = []
    scanned = 0
    excluded = 0
    for source_position, row in _iter_rows(path):
        scanned += 1
        if str(row.get("data_source")) in excluded_data_sources:
            excluded += 1
            continue
        priority = _priority(seed, domain, source_position)
        item = (-priority, -source_position, row)
        if len(heap) < rows:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != rows:
        raise ValueError(
            f"{domain}: requested {rows} rows but only retained {len(heap)} "
            f"after scanning {scanned} (excluded={excluded})"
        )
    selected = sorted(
        [(-neg_priority, -neg_position, row) for neg_priority, neg_position, row in heap]
    )
    records = [
        _normalize_record(
            row, domain=domain, source_position=source_position, mix_index=mix_index
        )
        for mix_index, (_, source_position, row) in enumerate(selected)
    ]
    return records, {
        "path": str(path),
        "source_rows": scanned,
        "selected_rows": len(records),
        "excluded_rows": excluded,
        "excluded_data_sources": sorted(excluded_data_sources),
    }


def _load_full_domain(
    path: Path, *, domain: str, excluded_data_sources: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scanned = 0
    excluded = 0
    for source_position, row in _iter_rows(path):
        scanned += 1
        if str(row.get("data_source")) in excluded_data_sources:
            excluded += 1
            continue
        records.append(
            _normalize_record(
                row,
                domain=domain,
                source_position=source_position,
                mix_index=len(records),
            )
        )
    return records, {
        "path": str(path),
        "source_rows": scanned,
        "selected_rows": len(records),
        "excluded_rows": excluded,
        "excluded_data_sources": sorted(excluded_data_sources),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--if-source", type=Path, required=True)
    parser.add_argument("--rows-per-domain", type=int, default=8000)
    parser.add_argument(
        "--full-union",
        action="store_true",
        help="Keep every non-excluded raw row; domain weighting is done by the training sampler.",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--allow-code-lcbv5", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    sources = {"math": args.math, "code": args.code, "if": args.if_source}
    selected: dict[str, list[dict[str, Any]]] = {}
    source_manifest: dict[str, Any] = {}
    for domain in DOMAINS:
        excluded = (
            set()
            if domain != "code" or args.allow_code_lcbv5
            else {"livecodebench"}
        )
        if args.full_union:
            selected[domain], source_manifest[domain] = _load_full_domain(
                sources[domain], domain=domain, excluded_data_sources=excluded
            )
        else:
            selected[domain], source_manifest[domain] = _select_domain(
                sources[domain],
                domain=domain,
                rows=args.rows_per_domain,
                seed=args.seed,
                excluded_data_sources=excluded,
            )

    if args.full_union:
        records = [record for domain in DOMAINS for record in selected[domain]]
    else:
        records = [
            selected[domain][index]
            for index in range(args.rows_per_domain)
            for domain in DOMAINS
        ]
    table = pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd", row_group_size=2048 if args.full_union else 192)

    manifest = {
        "contract": (
            "full raw non-farmed union; training-time weighted sampler"
            if args.full_union
            else "raw non-farmed domain pools; deterministic selection; strict math/code/if interleave"
        ),
        "seed": args.seed,
        "rows": table.num_rows,
        "rows_per_domain": {domain: len(rows) for domain, rows in selected.items()},
        "sources": source_manifest,
        "output": {"path": str(args.output), "sha256": _sha256(args.output)},
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

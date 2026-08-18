#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ifrl_family import IF_RL_SYSTEM_PROMPT, classify_family

DATA_SOURCE = "nemotron_if_rl"
ABILITY = "instruction_following"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val parquet files for Nemotron IF-RL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-train-parquet", type=Path, required=True)
    parser.add_argument("--output-val-parquet", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-stem",
        type=str,
        default="nemotron_if_rl",
        help="Base filename stem for train/val/summary outputs (e.g. nemotron_if_rl_train.parquet).",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, max_rows: int | None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                return
            yield json.loads(line)


def normalize_kwargs(kwargs: object) -> str:
    if kwargs is None:
        return "null"
    return json.dumps(kwargs, ensure_ascii=False, sort_keys=True)


def build_record(obj: dict) -> dict:
    prompt = obj["prompt"].strip()
    instruction_id_list = obj["instruction_id_list"]
    kwargs_list = obj["kwargs"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(instruction_id_list, list) or not instruction_id_list:
        raise ValueError("instruction_id_list must be a non-empty list")
    if not isinstance(kwargs_list, list) or len(kwargs_list) != len(instruction_id_list):
        raise ValueError("kwargs must align with instruction_id_list")

    agent_ref = obj.get("agent_ref", {})
    normalized_instruction_ids = [str(item) for item in instruction_id_list]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": IF_RL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "sample_id": int(obj["id"]),
            "raw_prompt": prompt,
            "instruction_id_list": normalized_instruction_ids,
            "instruction_kwargs_json": [normalize_kwargs(item) for item in kwargs_list],
            "dataset": str(obj.get("dataset", "")),
            "agent_name": str(agent_ref.get("name", "")),
            "family": classify_family(prompt),
        },
    }


def build_schema() -> pa.Schema:
    message_struct = pa.struct(
        [
            ("role", pa.string()),
            ("content", pa.string()),
        ]
    )
    reward_model_struct = pa.struct(
        [
            ("style", pa.string()),
            ("ground_truth", pa.string()),
        ]
    )
    extra_info_struct = pa.struct(
        [
            ("sample_id", pa.int64()),
            ("raw_prompt", pa.string()),
            ("instruction_id_list", pa.list_(pa.string())),
            ("instruction_kwargs_json", pa.list_(pa.string())),
            ("dataset", pa.string()),
            ("agent_name", pa.string()),
            ("family", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message_struct)),
            ("ability", pa.string()),
            ("reward_model", reward_model_struct),
            ("extra_info", extra_info_struct),
        ]
    )


def records_to_table(records: list[dict], schema: pa.Schema) -> pa.Table:
    frame = pd.DataFrame.from_records(records)
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


def write_parquet(path: Path, records: list[dict], batch_size: int, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    writer: pq.ParquetWriter | None = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        table = records_to_table(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(path, schema=schema)
        writer.write_table(table)
    if writer is None:
        writer = pq.ParquetWriter(path, schema=schema)
    writer.close()


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    raw_rows = 0
    kept_rows = 0
    filter_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    instruction_combo_counts: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    records: list[dict] = []

    for obj in iter_jsonl(args.input_jsonl, args.max_rows):
        raw_rows += 1
        try:
            record = build_record(obj)
        except (KeyError, TypeError, ValueError) as exc:
            filter_counts[str(exc)] += 1
            continue
        kept_rows += 1
        extra_info = record["extra_info"]
        dataset_counts[extra_info["dataset"]] += 1
        family_counts[extra_info["family"]] += 1
        instruction_combo_counts[len(extra_info["instruction_id_list"])] += 1
        instruction_counts.update(extra_info["instruction_id_list"])
        records.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_size = max(0, min(args.val_size, len(records)))
    val_records = records[:val_size]
    train_records = records[val_size:]

    schema = build_schema()
    write_parquet(args.output_train_parquet, train_records, batch_size=args.batch_size, schema=schema)

    output_val_parquet = args.output_val_parquet
    if output_val_parquet is None and val_size > 0:
        output_val_parquet = args.output_train_parquet.with_name(f"{args.output_stem}_val.parquet")
    if output_val_parquet is not None:
        write_parquet(output_val_parquet, val_records, batch_size=args.batch_size, schema=schema)

    summary_path = args.summary_path or args.output_train_parquet.with_suffix(".summary.json")
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_stem": args.output_stem,
        "system_prompt": IF_RL_SYSTEM_PROMPT,
        "output_train_parquet": str(args.output_train_parquet),
        "output_val_parquet": str(output_val_parquet) if output_val_parquet is not None else None,
        "summary_path": str(summary_path),
        "max_rows": args.max_rows,
        "batch_size": args.batch_size,
        "val_size": val_size,
        "seed": args.seed,
        "stats": {
            "raw_rows": raw_rows,
            "kept_rows": kept_rows,
            "filtered_rows": raw_rows - kept_rows,
            "train_rows": len(train_records),
            "val_rows": len(val_records),
        },
        "filter_counts": dict(sorted(filter_counts.items())),
        "instruction_combo_distribution": dict(sorted(instruction_combo_counts.items())),
        "top_instruction_ids": dict(instruction_counts.most_common(48)),
        "top_datasets": dict(dataset_counts.most_common(20)),
        "family_counts": dict(family_counts),
    }
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

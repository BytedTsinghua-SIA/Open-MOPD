#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import zlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert think_tag_sp LiveCodeBench v5/v6 fasteval parquets into Verl/OpenOPD code RL val parquet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-parquet",
        action="append",
        nargs=2,
        metavar=("SPLIT_NAME", "PATH"),
        required=True,
        help="Split name and local parquet path. Repeat for multiple inputs.",
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def resolve_input_path(path: str) -> Path:
    candidate = Path(path)
    if "://" in path:
        raise ValueError(f"remote URI is not supported: {path}")
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return candidate


def load_private_test_cases(raw: str) -> list[dict[str, Any]]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def build_ground_truth(metadata: dict[str, Any]) -> str:
    public = json.loads(metadata["public_test_cases"])
    private = load_private_test_cases(metadata["private_test_cases"])
    return json.dumps(public + private, ensure_ascii=False)


def build_schema() -> pa.Schema:
    message_struct = pa.struct([("role", pa.string()), ("content", pa.string())])
    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(message_struct)),
            ("ability", pa.string()),
            (
                "reward_model",
                pa.struct([("style", pa.string()), ("ground_truth", pa.string())]),
            ),
            (
                "extra_info",
                pa.struct(
                    [
                        ("split", pa.string()),
                        ("index", pa.string()),
                        ("sample_id", pa.string()),
                        ("dataset", pa.string()),
                        ("request_id", pa.string()),
                    ]
                ),
            ),
        ]
    )


def prompt_token_length(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    schema = build_schema()

    kept_rows: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    stats = {
        "output_parquet": str(args.output_parquet),
        "tokenizer_path": str(args.tokenizer_path),
        "max_prompt_length_filter": args.max_prompt_length,
        "sources": {},
        "filtered_overlong": 0,
        "filtered_errors": 0,
        "kept_rows": 0,
    }

    global_index = 0
    for split_name, input_path in args.input_parquet:
        parquet_path = resolve_input_path(input_path)
        source_stats = stats["sources"].setdefault(
            split_name,
            {
                "input_parquet": input_path,
                "raw_rows": 0,
                "kept_rows": 0,
                "filtered_overlong": 0,
                "filtered_errors": 0,
            },
        )
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            source_stats["raw_rows"] += 1
            try:
                messages = row["prompt"]
                token_len = prompt_token_length(tokenizer, messages)
                prompt_lengths.append(token_len)
                if args.max_prompt_length is not None and token_len > args.max_prompt_length:
                    source_stats["filtered_overlong"] += 1
                    stats["filtered_overlong"] += 1
                    continue
                metadata = json.loads(row["metadata"])
                kept_rows.append(
                    {
                        "data_source": split_name,
                        "prompt": messages,
                        "ability": "code",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": build_ground_truth(metadata),
                        },
                        "extra_info": {
                            "split": "val",
                            "index": str(global_index),
                            "sample_id": str(row.get("sample_id", "")),
                            "dataset": split_name,
                            "request_id": str(row.get("request_id", "")),
                        },
                    }
                )
                global_index += 1
                source_stats["kept_rows"] += 1
                stats["kept_rows"] += 1
            except Exception:
                source_stats["filtered_errors"] += 1
                stats["filtered_errors"] += 1

    stats["prompt_token_stats"] = {
        "count": len(prompt_lengths),
        "min": min(prompt_lengths) if prompt_lengths else None,
        "max": max(prompt_lengths) if prompt_lengths else None,
        "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None,
        "p50": percentile(prompt_lengths, 0.50),
        "p90": percentile(prompt_lengths, 0.90),
        "p99": percentile(prompt_lengths, 0.99),
        "recommended_max_prompt_length": max(prompt_lengths) if prompt_lengths else None,
    }

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(args.output_parquet, schema=schema) as writer:
        for start in range(0, len(kept_rows), args.batch_size):
            batch = kept_rows[start : start + args.batch_size]
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))

    summary_path = args.summary_path or args.output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

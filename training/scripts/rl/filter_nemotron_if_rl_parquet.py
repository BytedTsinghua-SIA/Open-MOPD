#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_nemotron_if_rl_dataset import build_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Nemotron IF-RL parquet rows by prompt token length."
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def prompt_token_length(tokenizer, messages: list[dict], apply_chat_template_kwargs: dict) -> int:
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs,
    )
    return len(tokenizer(raw_prompt, add_special_tokens=False)["input_ids"])


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    apply_kwargs = {"enable_thinking": False}
    schema = build_schema()

    table = pq.read_table(args.input_parquet)
    kept_rows = []
    filtered_overlong = 0
    token_lengths = []

    for row in table.to_pylist():
        token_len = prompt_token_length(tokenizer, row["prompt"], apply_kwargs)
        token_lengths.append(token_len)
        if token_len > args.max_prompt_length:
            filtered_overlong += 1
            continue
        kept_rows.append(row)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(args.output_parquet, schema=schema) as writer:
        for start in range(0, len(kept_rows), args.batch_size):
            batch = kept_rows[start : start + args.batch_size]
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))

    summary = {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "tokenizer_path": str(args.tokenizer_path),
        "max_prompt_length": args.max_prompt_length,
        "apply_chat_template_kwargs": apply_kwargs,
        "raw_rows": table.num_rows,
        "kept_rows": len(kept_rows),
        "filtered_overlong": filtered_overlong,
        "prompt_token_stats": {
            "min": min(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
            "mean": sum(token_lengths) / len(token_lengths) if token_lengths else None,
        },
    }
    summary_path = args.summary_path or args.output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

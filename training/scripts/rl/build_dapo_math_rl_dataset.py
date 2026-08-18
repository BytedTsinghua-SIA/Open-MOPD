#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
MATH_SCRIPT_DIR = SCRIPT_DIR.parent / "math"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(MATH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MATH_SCRIPT_DIR))

from math_rl_prompts import MATH_RL_THINK_TAG_SYSTEM_PROMPT, MATH_RL_USER_SUFFIX
from normalize_verl_math_data import as_messages, strip_old_math_instruction

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DAPO math-17k RL parquet with 20260608 think_tag_sp system prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def extract_user_content(prompt_value: Any) -> str:
    messages = as_messages(prompt_value)
    user_content = ""
    for message in messages:
        if message["role"] == "user":
            user_content = message["content"]
            break
    if not user_content and messages:
        user_content = messages[-1]["content"]
    return strip_old_math_instruction(user_content)


def build_prompt_messages(user_content: str) -> list[dict[str, str]]:
    user = f"{user_content}\n\n{MATH_RL_USER_SUFFIX}"
    return [
        {"role": "system", "content": MATH_RL_THINK_TAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def iter_rows(path: Path, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


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
            ("extra_info", pa.struct([("index", pa.string())])),
        ]
    )


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema=schema) as writer:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def main() -> None:
    args = parse_args()
    schema = build_schema()
    kept_rows = 0
    output_rows: list[dict[str, Any]] = []

    for batch in iter_rows(args.input_parquet, args.batch_size):
        for row in batch:
            if args.max_rows is not None and kept_rows >= args.max_rows:
                break
            reward_model = row.get("reward_model") or {}
            extra_info = row.get("extra_info") or {}
            user_content = extract_user_content(row.get("prompt"))
            if not user_content:
                continue
            output_rows.append(
                {
                    "data_source": str(row.get("data_source") or "math_dapo_boxed"),
                    "prompt": build_prompt_messages(user_content),
                    "ability": str(row.get("ability") or "MATH"),
                    "reward_model": {
                        "style": str(reward_model.get("style") or "rule"),
                        "ground_truth": str(reward_model.get("ground_truth") or ""),
                    },
                    "extra_info": {"index": str(extra_info.get("index") or "")},
                }
            )
            kept_rows += 1
        if args.max_rows is not None and kept_rows >= args.max_rows:
            break

    write_parquet(args.output_parquet, output_rows, schema)

    summary_path = args.summary_path or args.output_parquet.with_suffix(".summary.json")
    summary = {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "system_prompt": MATH_RL_THINK_TAG_SYSTEM_PROMPT,
        "user_suffix": MATH_RL_USER_SUFFIX,
        "kept_rows": kept_rows,
        "max_rows": args.max_rows,
        "batch_size": args.batch_size,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

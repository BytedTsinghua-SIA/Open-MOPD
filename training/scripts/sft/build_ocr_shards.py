from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenCodeReasoning into sharded SFT parquet files.")
    parser.add_argument("--ocr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assistant-key", choices=["output", "solution"], default="output")
    parser.add_argument("--prompt-style", choices=["plain", "lcb"], default="plain")
    parser.add_argument("--include-splits", nargs="+", choices=["split_0", "split_1"], default=["split_0"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--shard-rows", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_messages(user_text: str, assistant_text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]


def convert_frame(frame: pd.DataFrame, assistant_key: str, source_split: str) -> pd.DataFrame:
    required = {"id", "input", assistant_key, "dataset", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    converted = pd.DataFrame(
        {
            "id": frame["id"].astype("string").tolist(),
            "messages": [
                build_messages(str(user_text), str(assistant_text))
                for user_text, assistant_text in zip(
                    frame["input"].fillna("").astype("string").tolist(),
                    frame[assistant_key].fillna("").astype("string").tolist(),
                    strict=False,
                )
            ],
            "source": frame["source"].astype("string").tolist() if "source" in frame.columns else [None] * len(frame),
            "dataset": frame["dataset"].astype("string").tolist(),
            "split": frame["split"].astype("string").tolist(),
            "difficulty": frame["difficulty"].astype("string").tolist() if "difficulty" in frame.columns else [None] * len(frame),
            "ocr_split": [source_split] * len(frame),
        }
    )
    converted = converted[converted["messages"].map(lambda x: bool(x[0]["content"].strip()) and bool(x[1]["content"].strip()))]
    return converted.reset_index(drop=True)


def iter_input_files(ocr_root: Path, include_splits: list[str]) -> list[tuple[str, Path]]:
    results = []
    for split_name in include_splits:
        files = sorted((ocr_root / split_name).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {ocr_root / split_name}")
        results.extend((split_name, path) for path in files)
    return results


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.output_dir.glob("train-*.parquet"):
        old.unlink()

    buffer = []
    shard_index = 0
    total_rows = 0
    remaining = args.max_samples
    schema = None

    for split_name, parquet_file in iter_input_files(args.ocr_root, args.include_splits):
        if remaining == 0:
            break
        frame = pd.read_parquet(parquet_file)
        converted = convert_frame(frame, assistant_key=args.assistant_key, source_split=split_name)
        if remaining >= 0:
            converted = converted.iloc[:remaining]
            remaining -= len(converted)
        if converted.empty:
            continue
        records = converted.to_dict("records")
        for record in records:
            buffer.append(record)
            if len(buffer) >= args.shard_rows:
                table = pa.Table.from_pylist(buffer, schema=schema)
                if schema is None:
                    schema = table.schema
                pq.write_table(table, args.output_dir / f"train-{shard_index:05d}.parquet")
                total_rows += len(buffer)
                print(f"wrote shard {shard_index} rows={len(buffer)}")
                shard_index += 1
                buffer = []
    if buffer:
        table = pa.Table.from_pylist(buffer, schema=schema)
        if schema is None:
            schema = table.schema
        pq.write_table(table, args.output_dir / f"train-{shard_index:05d}.parquet")
        total_rows += len(buffer)
        print(f"wrote shard {shard_index} rows={len(buffer)}")

    summary = {
        "total_rows": total_rows,
        "num_shards": len(list(args.output_dir.glob('train-*.parquet'))),
        "assistant_key": args.assistant_key,
        "prompt_style": args.prompt_style,
        "include_splits": args.include_splits,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

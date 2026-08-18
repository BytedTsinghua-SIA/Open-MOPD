from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SHARD_ROWS = 50_000
DEFAULT_WRITE_BATCH_ROWS = 1024
OUTPUT_FIELDS = ["messages", "domain", "source", "dataset", "split", "difficulty", "sample_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFT parquet input into pandas-readable train shards.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file or directory of *.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", type=str, default="")
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--write-batch-rows", type=int, default=DEFAULT_WRITE_BATCH_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no *.parquet files found under {path}")
        return files
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def normalize_record(row: dict[str, Any], domain: str) -> dict[str, Any]:
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    normalized_messages = []
    for message in messages or []:
        normalized_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )

    sample_id = row.get("id")
    if sample_id is None:
        sample_id = row.get("sample_id")
    if sample_id is None:
        sample_id = row.get("source_row_index")
    if sample_id is None:
        sample_id = row.get("key")

    source = row.get("source")
    dataset = row.get("dataset", domain)
    split = row.get("split", row.get("ocr_split", ""))
    difficulty = row.get("difficulty", "")

    return {
        "messages": normalized_messages,
        "domain": str(domain),
        "source": "" if source is None else str(source),
        "dataset": "" if dataset is None else str(dataset),
        "split": "" if split is None else str(split),
        "difficulty": "" if difficulty is None else str(difficulty),
        "sample_id": "" if sample_id is None else str(sample_id),
    }


def load_rows(path: Path, domain: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_file in iter_input_files(path):
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                rows.append(normalize_record(row, domain))
    return rows


def write_shards(
    rows: list[dict[str, Any]],
    output_dir: Path,
    shard_rows: int,
    write_batch_rows: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("train-*.parquet"):
        old.unlink()

    schema = pa.schema(
        [
            ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("domain", pa.string()),
            ("source", pa.string()),
            ("dataset", pa.string()),
            ("split", pa.string()),
            ("difficulty", pa.string()),
            ("sample_id", pa.string()),
        ]
    )

    shard_paths: list[str] = []
    for start in range(0, len(rows), shard_rows):
        shard_path = output_dir / f"train-{start // shard_rows:05d}.parquet"
        writer: pq.ParquetWriter | None = None
        try:
            shard_end = min(start + shard_rows, len(rows))
            for batch_start in range(start, shard_end, write_batch_rows):
                batch_end = min(batch_start + write_batch_rows, shard_end)
                batch_rows = rows[batch_start:batch_end]
                table = pa.Table.from_pylist(batch_rows, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(shard_path, schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        pd.read_parquet(shard_path)
        shard_paths.append(str(shard_path))
    return shard_paths


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")

    rows = load_rows(args.input, args.domain)
    shard_paths = write_shards(rows, args.output_dir, args.shard_rows, args.write_batch_rows)
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "domain": args.domain,
        "rows": len(rows),
        "num_shards": len(shard_paths),
        "shard_rows": args.shard_rows,
        "write_batch_rows": args.write_batch_rows,
        "shards": shard_paths,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

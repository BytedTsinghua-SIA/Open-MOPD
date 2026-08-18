from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SHARD_ROWS = 50_000
DEFAULT_WRITE_BATCH_ROWS = 1024
OUTPUT_FIELDS = ["messages", "domain", "source", "dataset", "split", "difficulty", "sample_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weighted SFT parquet mix from source parquets.")
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "DOMAIN", "PARQUET"), required=True)
    parser.add_argument("--weight", action="append", nargs=2, metavar=("NAME", "WEIGHT"), required=True)
    parser.add_argument("--target", action="append", nargs=2, metavar=("NAME", "ROWS"), help="Per-source target rows.")
    parser.add_argument("--total-rows", type=int, help="Fallback total rows when per-source targets are not provided.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    weights = {name: int(weight) for name, weight in args.weight}
    targets = {name: int(rows) for name, rows in (args.target or [])}
    result = []
    for name, domain, parquet in args.source:
        if name not in weights:
            raise ValueError(f"missing weight for source {name}")
        if weights[name] <= 0:
            raise ValueError(f"weight for source {name} must be positive")
        result.append(
            {
                "name": name,
                "domain": domain,
                "path": Path(parquet),
                "weight": weights[name],
                "target_rows": targets.get(name),
            }
        )
    return result


def read_rows(path: Path, domain: str) -> list[dict[str, Any]]:
    def load_one_file(parquet_file: Path) -> list[dict[str, Any]]:
        pf = pq.ParquetFile(parquet_file)
        loaded: list[dict[str, Any]] = []
        for batch in pf.iter_batches(batch_size=2048):
            loaded.extend(batch.to_pylist())
        return loaded

    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"no *.parquet files found under {path}")
        rows = []
        for parquet_file in parquet_files:
            rows.extend(load_one_file(parquet_file))
    else:
        rows = load_one_file(path)
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, str):
            try:
                row["messages"] = json.loads(messages)
            except json.JSONDecodeError:
                pass
    return [normalize_record(row, domain) for row in rows]


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


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows but only {len(rows)} available")
    indices = random.Random(seed).sample(range(len(rows)), count)
    return [rows[i] for i in indices]


def validate_parquet_readable(path: Path) -> None:
    import pandas as pd

    pd.read_parquet(path)


def write_shards(
    rows: list[dict[str, Any]],
    output_dir: Path,
    shard_rows: int,
    write_batch_rows: int = DEFAULT_WRITE_BATCH_ROWS,
) -> None:
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
        validate_parquet_readable(shard_path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is non-empty; pass --overwrite")

    sources = parse_sources(args)
    use_explicit_targets = any(source["target_rows"] is not None for source in sources)
    if use_explicit_targets:
        if not all(source["target_rows"] is not None for source in sources):
            raise ValueError("either specify --target for every source or for none of them")
        allocations = [(source, int(source["target_rows"])) for source in sources]
    else:
        if args.total_rows is None:
            raise ValueError("--total-rows is required when per-source targets are not provided")
        weight_sum = sum(source["weight"] for source in sources)
        allocations = []
        consumed = 0
        for index, source in enumerate(sources):
            rows = args.total_rows * source["weight"] // weight_sum
            if index == len(sources) - 1:
                rows = args.total_rows - consumed
            consumed += rows
            allocations.append((source, rows))

    mixed = []
    pools = []
    for index, (source, rows) in enumerate(allocations):
        pools.append(
            sample_rows(
                read_rows(source["path"], source["domain"]),
                rows,
                args.seed + (index + 1) * 111,
            )
        )
    while any(pools):
        for pool in pools:
            if pool:
                mixed.append(pool.pop())

    write_shards(mixed, args.output_dir, args.shard_rows)
    summary = {
        "sources": [
            {
                "name": source["name"],
                "domain": source["domain"],
                "path": str(source["path"]),
                "weight": source["weight"],
                "rows": rows,
            }
            for source, rows in allocations
        ],
        "rows": {"total": len(mixed)},
        "mode": "explicit_targets" if use_explicit_targets else "weights_plus_total_rows",
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

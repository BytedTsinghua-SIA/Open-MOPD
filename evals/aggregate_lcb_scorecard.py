"""Aggregate local LiveCodeBench score files into a per-model scorecard."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODELS = {
    "codesft_start": "code SFT (RL start)",
    "d40": "0623d S40 (rcorr2.0)",
    "d80": "0623d S80",
    "e40": "0623e S40 (norcorr)",
    "e80": "0623e S80",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="local eval output root")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    aggregate: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    shard_counts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for model in MODELS:
        for path in sorted(args.root.glob(f"20260624_{model}_lcb_sharded/**/scores_livecodebench_v*.json")):
            dataset = "v5" if "_v5" in path.name else "v6"
            values = json.loads(path.read_text(encoding="utf-8"))
            aggregate[model][dataset][0] += int(values.get("correct", 0))
            aggregate[model][dataset][1] += int(values.get("scored_rows", 0))
            shard_counts[model][dataset].add(path.parent.name)

    rows: dict[str, dict[str, object]] = {}
    print(f"{'model':24} {'LCB v5':>16} {'LCB v6':>16} {'code avg':>9}  shards")
    print("-" * 82)
    for model, name in MODELS.items():
        if model not in aggregate:
            continue
        v5c, v5n = aggregate[model]["v5"]
        v6c, v6n = aggregate[model]["v6"]
        p5 = 100 * v5c / v5n if v5n else 0.0
        p6 = 100 * v6c / v6n if v6n else 0.0
        code_avg = (p5 + p6) / 2 if v5n and v6n else (p5 or p6)
        ns5 = len(shard_counts[model]["v5"])
        ns6 = len(shard_counts[model]["v6"])
        print(f"{name:24} {p5:6.2f}% ({v5c:4d}/{v5n:4d}) {p6:6.2f}% ({v6c:4d}/{v6n:4d}) {code_avg:7.2f}%  v5:{ns5} v6:{ns6}")
        rows[model] = {
            "name": name,
            "v5_pct": p5,
            "v6_pct": p6,
            "code_avg": code_avg,
            "v5": [v5c, v5n],
            "v6": [v6c, v6n],
            "shards": [ns5, ns6],
        }
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

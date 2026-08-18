"""Deterministic domain-weighted sampler for mixed RL training."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sized

from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler


DOMAIN_ORDER = ("math", "code", "if")


def weighted_quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Hamilton allocation with stable domain-order tie breaking."""

    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    normalized = {domain: float(weights[domain]) for domain in DOMAIN_ORDER}
    if any(weight <= 0 for weight in normalized.values()):
        raise ValueError(f"domain weights must be positive: {normalized}")
    scale = sum(normalized.values())
    raw = {domain: total * weight / scale for domain, weight in normalized.items()}
    quotas = {domain: math.floor(value) for domain, value in raw.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(DOMAIN_ORDER, key=lambda domain: (-(raw[domain] - quotas[domain]), DOMAIN_ORDER.index(domain)))
    for domain in priority[:remainder]:
        quotas[domain] += 1
    return quotas


class DomainWeightedSampler(AbstractSampler):
    """Yield exact weighted domain mixtures in every generation batch.

    Each domain pool is shuffled independently and cycled when exhausted, so
    small domains are oversampled without discarding any source rows.  The
    emitted epoch length is the full union length rounded down to full batches.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))
        self.seed = int(data_config.get("seed", 0) or 0)
        configured = data_config.get("domain_weights", {"math": 1, "code": 1, "if": 1})
        self.weights = {domain: float(configured[domain]) for domain in DOMAIN_ORDER}
        self.quotas = weighted_quotas(self.batch_size, self.weights)
        self.num_samples = (len(data_source) // self.batch_size) * self.batch_size
        if self.num_samples <= 0:
            raise ValueError(
                f"dataset length {len(data_source)} is smaller than weighted batch size {self.batch_size}"
            )

        dataframe = getattr(data_source, "dataframe", None)
        if dataframe is None or "domain" not in dataframe.column_names:
            raise ValueError("DomainWeightedSampler requires a top-level dataset 'domain' column")
        pools: dict[str, list[int]] = defaultdict(list)
        for index, domain_value in enumerate(dataframe["domain"]):
            pools[str(domain_value)].append(index)
        self.pools = {domain: pools[domain] for domain in DOMAIN_ORDER}
        missing = [domain for domain, indices in self.pools.items() if not indices]
        if missing:
            raise ValueError(f"weighted sampler has empty domain pools: {missing}")
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {domain: list(indices) for domain, indices in self.pools.items()}
        cursors = {domain: 0 for domain in DOMAIN_ORDER}
        for indices in pools.values():
            rng.shuffle(indices)

        def take(domain: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                pool = pools[domain]
                cursor = cursors[domain]
                remaining = len(pool) - cursor
                amount = min(count - len(selected), remaining)
                selected.extend(pool[cursor : cursor + amount])
                cursors[domain] += amount
                if cursors[domain] == len(pool):
                    rng.shuffle(pool)
                    cursors[domain] = 0
            return selected

        num_batches = self.num_samples // self.batch_size
        for batch_index in range(num_batches):
            # Alternate an indivisible tie between Math and Code so long-run
            # exposure remains exactly symmetric for 2:2:1.
            quotas = dict(self.quotas)
            if (
                batch_index % 2 == 1
                and self.weights["math"] == self.weights["code"]
                and quotas["math"] == quotas["code"] + 1
            ):
                quotas["math"], quotas["code"] = quotas["code"], quotas["math"]
            batch: list[int] = []
            for domain in DOMAIN_ORDER:
                batch.extend(take(domain, quotas[domain]))
            rng.shuffle(batch)
            yield from batch

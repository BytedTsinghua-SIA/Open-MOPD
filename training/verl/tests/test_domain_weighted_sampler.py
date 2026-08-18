from collections import Counter
from itertools import islice

from omegaconf import OmegaConf

from verl.utils.dataset.domain_weighted_sampler import DomainWeightedSampler, weighted_quotas


class _Frame:
    column_names = ["domain"]

    def __init__(self, domains):
        self.domains = domains

    def __getitem__(self, key):
        assert key == "domain"
        return self.domains


class _Dataset:
    def __init__(self):
        self.domains = ["math"] * 20 + ["code"] * 30 + ["if"] * 50
        self.dataframe = _Frame(self.domains)

    def __len__(self):
        return len(self.domains)


def test_weighted_quotas_2to2to1() -> None:
    weights = {"math": 2, "code": 2, "if": 1}
    assert weighted_quotas(192, weights) == {"math": 77, "code": 77, "if": 38}
    assert weighted_quotas(576, weights) == {"math": 231, "code": 230, "if": 115}


def test_sampler_emits_exact_weighted_batches_and_cycles_small_pools() -> None:
    dataset = _Dataset()
    config = OmegaConf.create(
        {
            "train_batch_size": 10,
            "gen_batch_size": 10,
            "seed": 7,
            "domain_weights": {"math": 2, "code": 2, "if": 1},
        }
    )
    sampler = DomainWeightedSampler(dataset, config)
    indices = list(islice(iter(sampler), 20))
    first = Counter(dataset.domains[index] for index in indices[:10])
    second = Counter(dataset.domains[index] for index in indices[10:20])

    assert first == {"math": 4, "code": 4, "if": 2}
    assert second == first
    assert len(sampler) == 100

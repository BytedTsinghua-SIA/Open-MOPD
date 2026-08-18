import pytest

from verl.workers.rollout.seed import ranked_rollout_seed


def test_rollout_replicas_receive_distinct_reproducible_engine_seeds():
    assert [ranked_rollout_seed(17, rank) for rank in range(4)] == [17, 18, 19, 20]


def test_negative_rollout_rank_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ranked_rollout_seed(0, -1)

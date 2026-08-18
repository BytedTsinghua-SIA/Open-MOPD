import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

VERL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERL_ROOT))

from verl.experimental.reward.reward_loop.mixed import _score_math_in_process  # noqa: E402
from verl.utils.reward_score.mix_rl_dispatch import infer_domain  # noqa: E402
from verl.workers.reward_manager import get_reward_manager_cls  # noqa: E402
from verl.workers.reward_manager.mixed import MixedRewardManager  # noqa: E402


def test_mix_domain_dispatch_covers_training_and_validation_sources() -> None:
    assert infer_domain("math_dapo_boxed") == "math"
    assert infer_domain("aime24") == "math"
    assert infer_domain("primeintellect") == "code"
    assert infer_domain("taco") == "code"
    assert infer_domain("livecodebench_v6") == "code"
    assert infer_domain("nemotron_if_rl") == "if"
    assert infer_domain("ifeval") == "if"
    assert infer_domain("ifbench_mt_ifbench") == "if"


def test_explicit_domain_takes_precedence() -> None:
    assert infer_domain("custom", {"domain": "code"}, "instruction_following") == "code"


def test_mixed_batch_reward_manager_is_registered_for_trainer_startup() -> None:
    assert get_reward_manager_cls("mixed") is MixedRewardManager


def test_math_scorer_keeps_signal_timeout_inside_spawn_process() -> None:
    response = "<think>done</think> The answer is \\boxed{2}."
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = pool.submit(_score_math_in_process, "aime24", response, "2", {}).result(timeout=30)
    assert result["score"] == 1.0
    assert result["acc"] is True

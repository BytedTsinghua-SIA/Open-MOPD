import copy

from omegaconf import OmegaConf

from verl.trainer.main_ppo import _effective_rl_dataset_config


def _config():
    return OmegaConf.create(
        {
            "max_prompt_length": 2048,
            "val_max_prompt_length": 20000,
            "filter_overlong_prompts": True,
            "val_filter_overlong_prompts": False,
            "truncation": "right",
            "val_truncation": "error",
        }
    )


def test_training_dataset_keeps_training_limits_and_identity():
    config = _config()
    assert _effective_rl_dataset_config(config, is_train=True) is config
    assert config.max_prompt_length == 2048


def test_validation_dataset_uses_independent_limits_without_mutation():
    config = _config()
    original = copy.deepcopy(config)
    effective = _effective_rl_dataset_config(config, is_train=False)

    assert effective is not config
    assert effective.max_prompt_length == 20000
    assert effective.filter_overlong_prompts is False
    assert effective.truncation == "error"
    assert config == original


def test_validation_dataset_falls_back_when_overrides_are_null():
    config = _config()
    config.val_max_prompt_length = None
    config.val_filter_overlong_prompts = None
    config.val_truncation = None
    effective = _effective_rl_dataset_config(config, is_train=False)

    assert effective.max_prompt_length == 2048
    assert effective.filter_overlong_prompts is True
    assert effective.truncation == "right"

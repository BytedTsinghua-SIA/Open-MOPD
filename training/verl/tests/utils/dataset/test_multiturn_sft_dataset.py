from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class _RecordingTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


def _dataset(apply_chat_template_kwargs=None):
    dataset = object.__new__(MultiTurnSFTDataset)
    dataset.tokenizer = _RecordingTokenizer()
    dataset.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
    return dataset


def test_omits_unspecified_enable_thinking_to_preserve_template_default():
    dataset = _dataset()

    dataset._apply_chat_template(
        [], tokenize=False, add_generation_prompt=False, enable_thinking=None, tools=None
    )

    assert "enable_thinking" not in dataset.tokenizer.kwargs


def test_uses_spec_level_enable_thinking_when_row_does_not_override_it():
    dataset = _dataset({"enable_thinking": False})

    dataset._apply_chat_template(
        [], tokenize=False, add_generation_prompt=False, enable_thinking=None, tools=None
    )

    assert dataset.tokenizer.kwargs["enable_thinking"] is False


def test_row_enable_thinking_overrides_spec_level_value_without_duplicate_kwarg():
    dataset = _dataset({"enable_thinking": False})

    dataset._apply_chat_template(
        [], tokenize=False, add_generation_prompt=False, enable_thinking=True, tools=None
    )

    assert dataset.tokenizer.kwargs["enable_thinking"] is True

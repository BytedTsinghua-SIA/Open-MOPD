from types import SimpleNamespace

from transformers import AutoModelForCausalLM

from verl.utils.model import get_hf_auto_model_class


def test_conditional_generation_architecture_uses_causal_lm_auto_class():
    config = SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"])

    assert get_hf_auto_model_class(config) is AutoModelForCausalLM

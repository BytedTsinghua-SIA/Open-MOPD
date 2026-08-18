import pytest
import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class PresentDecoderLayer(nn.Module):
    pass


class CompositeTextModel(nn.Module):
    _no_split_modules = ["PresentDecoderLayer", "OptionalVisionLayer"]

    def __init__(self):
        super().__init__()
        self.layer = PresentDecoderLayer()


def test_model_default_wrap_policy_skips_absent_optional_components():
    policy = get_fsdp_wrap_policy(CompositeTextModel(), config={"min_num_params": 0})

    assert policy is not None


def test_explicit_wrap_policy_rejects_absent_class():
    with pytest.raises(Exception, match="MissingDecoderLayer"):
        get_fsdp_wrap_policy(
            CompositeTextModel(),
            config={"min_num_params": 0, "transformer_layer_cls_to_wrap": ["MissingDecoderLayer"]},
        )

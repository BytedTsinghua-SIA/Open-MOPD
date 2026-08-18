import pytest
import torch

from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _PACKED_WEIGHT_SYNC_PROBES,
    _WEIGHT_SYNC_PROBE_NAMES,
    _load_weights_with_optional_verification,
)


class _ProbeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(4, elementwise_affine=True, bias=False, dtype=torch.bfloat16)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class _ProbeModel(torch.nn.Module):
    def __init__(self, *, skip_name: str | None = None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.model.layers = torch.nn.ModuleList([_ProbeLayer()])
        self.model.norm = torch.nn.LayerNorm(4, elementwise_affine=True, bias=False, dtype=torch.bfloat16)
        self.skip_name = skip_name

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name != self.skip_name:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


def _source_weights():
    shapes = {
        "model.embed_tokens.weight": (8, 4),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.mlp.down_proj.weight": (4, 4),
        "model.norm.weight": (4,),
    }
    return [
        (name, torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape))
        for name, shape in shapes.items()
    ]


def test_weight_sync_verification_accepts_matching_direct_weights():
    model = _ProbeModel()
    loaded = _load_weights_with_optional_verification(model, iter(_source_weights()), enabled=True)
    assert loaded == set(_WEIGHT_SYNC_PROBE_NAMES)


def test_weight_sync_verification_rejects_silently_skipped_weight():
    skipped = "model.norm.weight"
    model = _ProbeModel(skip_name=skipped)
    with pytest.raises(RuntimeError, match="weight synchronization verification failed"):
        _load_weights_with_optional_verification(model, iter(_source_weights()), enabled=True)


class _PackedProbeModel(_ProbeModel):
    def __init__(self):
        super().__init__()
        self.model.layers[0].self_attn = torch.nn.Module()
        self.model.layers[0].self_attn.qkv_proj = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
        self.model.layers[0].mlp.gate_up_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        packed = {
            "model.layers.0.self_attn.q_proj.weight": ("model.layers.0.self_attn.qkv_proj.weight", 0, 2),
            "model.layers.0.self_attn.k_proj.weight": ("model.layers.0.self_attn.qkv_proj.weight", 2, 2),
            "model.layers.0.self_attn.v_proj.weight": ("model.layers.0.self_attn.qkv_proj.weight", 4, 2),
            "model.layers.0.mlp.gate_proj.weight": ("model.layers.0.mlp.gate_up_proj.weight", 0, 2),
            "model.layers.0.mlp.up_proj.weight": ("model.layers.0.mlp.gate_up_proj.weight", 2, 2),
        }
        for name, tensor in weights:
            if name in packed:
                target_name, offset, rows = packed[name]
                params[target_name].data.narrow(0, offset, rows).copy_(tensor)
                loaded.add(target_name)
            else:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


def test_weight_sync_verification_accepts_matching_packed_weights(capsys):
    model = _PackedProbeModel()
    packed_shapes = {
        name: (2, 4)
        for name in _PACKED_WEIGHT_SYNC_PROBES
    }
    packed_weights = [
        (name, torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape))
        for name, shape in packed_shapes.items()
    ]
    loaded = _load_weights_with_optional_verification(
        model,
        iter([*_source_weights(), *packed_weights]),
        enabled=True,
    )
    assert "model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.layers.0.mlp.gate_up_proj.weight" in loaded
    assert "OPENOPD_WEIGHT_SYNC_VERIFY" in capsys.readouterr().out


class _LoadedButUninspectablePackedProbeModel(_ProbeModel):
    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in _PACKED_WEIGHT_SYNC_PROBES:
                # Mirrors vLLM implementations that acknowledge the original
                # HF name without exposing their packed storage parameter.
                loaded.add(name)
            else:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


def test_weight_sync_verification_accepts_loader_acknowledged_uninspectable_packed_weights(capsys):
    model = _LoadedButUninspectablePackedProbeModel()
    packed_weights = [
        (name, torch.ones((2, 4), dtype=torch.float32))
        for name in _PACKED_WEIGHT_SYNC_PROBES
    ]
    _load_weights_with_optional_verification(
        model,
        iter([*_source_weights(), *packed_weights]),
        enabled=True,
    )
    assert "loaded_uninspectable" in capsys.readouterr().out


class _SkippedUninspectablePackedProbeModel(_LoadedButUninspectablePackedProbeModel):
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        loaded.discard("model.layers.0.self_attn.q_proj.weight")
        return loaded


def test_weight_sync_verification_rejects_unacknowledged_uninspectable_packed_weight():
    model = _SkippedUninspectablePackedProbeModel()
    packed_weights = [
        (name, torch.ones((2, 4), dtype=torch.float32))
        for name in _PACKED_WEIGHT_SYNC_PROBES
    ]
    with pytest.raises(RuntimeError, match="weight synchronization verification failed"):
        _load_weights_with_optional_verification(
            model,
            iter([*_source_weights(), *packed_weights]),
            enabled=True,
        )

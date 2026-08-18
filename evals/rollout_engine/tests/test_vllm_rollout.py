from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from evals.rollout_engine.vllm_rollout import (
    llm_kwargs,
    resolve_stop_token_ids,
    run_file_partition,
    sampling_from_config,
)


class _FakeOutput:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


class _FakeRequestOutput:
    def __init__(self, outputs: list[_FakeOutput]) -> None:
        self.outputs = outputs


class _FakeTokenizer:
    all_special_ids = [4]

    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT:" + str(prompt)


class _FakeLLM:
    def __init__(self) -> None:
        self._tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, params):
        assert len(prompts) == 1
        assert params.n == 2
        return [
            _FakeRequestOutput(
                outputs=[
                    _FakeOutput("first", [1, 2, 3]),
                    _FakeOutput("second", [4, 5]),
                ]
            )
        ]


def test_llm_kwargs_keeps_data_parallelism_process_local() -> None:
    args = SimpleNamespace(
        model="/tmp/model",
        tensor_parallel_size=1,
        data_parallel_size=8,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.93,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_num_seqs=256,
        max_num_batched_tokens=None,
    )
    kwargs = llm_kwargs(args)
    assert "data_parallel_size" not in kwargs


def test_sampling_from_config_supports_n() -> None:
    params = sampling_from_config(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        n=3,
        max_tokens=64,
        stop_token_ids=[151643, 151645],
    )
    assert params.n == 3
    assert params.top_p == 0.95
    assert params.top_k == 20
    assert params.max_tokens == 64


def test_resolve_stop_token_ids_loads_generation_config(tmp_path: Path) -> None:
    from transformers import GenerationConfig

    GenerationConfig(eos_token_id=[151645]).save_pretrained(tmp_path)
    assert resolve_stop_token_ids(str(tmp_path), None) == [151645]


def test_resolve_stop_token_ids_prefers_explicit_ids(tmp_path: Path) -> None:
    assert resolve_stop_token_ids(str(tmp_path), "151645,151643") == [151645, 151643]


def test_run_file_flattens_multiple_completions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "outputs"

    pd.DataFrame(
        [
            {
                "dataset": "aime24",
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Solve 1+1."},
                ],
            }
        ]
    ).to_parquet(input_path, index=False)

    out_paths = run_file_partition(
        llm=_FakeLLM(),
        path=input_path,
        output_dir=output_dir,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        n=2,
        max_tokens=64,
        stop_token_ids=None,
        enable_thinking=True,
        dp_size=1,
        global_dp_rank=0,
        base=0,
        offset=None,
    )

    assert len(out_paths) == 1
    out_df = pd.read_parquet(out_paths[0]).sort_values("completion_index").reset_index(drop=True)
    assert len(out_df) == 2
    assert out_df["completion_index"].tolist() == [0, 1]
    assert out_df["completion"].tolist() == ["first", "second"]
    assert out_df["completion_tokens"].tolist() == [3, 2]
    assert out_df["finish_reason"].isna().all()
    assert out_df["stop_reason"].isna().all()
    assert out_df["generated_special_token_counts"].tolist() == ["{}", '{"4": 1}']
    assert out_df["top_p"].tolist() == [0.95, 0.95]
    assert out_df["top_k"].tolist() == [20, 20]
    assert out_df["max_tokens"].tolist() == [64, 64]

#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate that MixSFT messages survive model-family chat templates unchanged."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig

from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def parse_mapping(value: str) -> tuple[str, str]:
    name, separator, payload = value.partition("=")
    if not separator or not name or not payload:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Normalized MixSFT parquet produced by build_weighted_mix.py")
    parser.add_argument("--tokenizer", action="append", type=parse_mapping, required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--template-kwargs",
        action="append",
        type=parse_mapping,
        default=[],
        metavar='NAME={"key":"value"}',
    )
    parser.add_argument(
        "--expected-generation-eos",
        action="append",
        type=parse_mapping,
        default=[],
        metavar="NAME=ID,ID",
    )
    parser.add_argument("--max-length", type=int, default=32768)
    return parser.parse_args()


def assert_turn_boundaries(name: str, rendered: str) -> None:
    boundary_pairs = (
        ("<|im_start|>", "<|im_end|>\n"),
        ("<|start_header_id|>", "<|eot_id|>"),
        ("<start_of_turn>", "<end_of_turn>\n"),
    )
    for start_marker, previous_end in boundary_pairs:
        starts = []
        offset = 0
        while (position := rendered.find(start_marker, offset)) >= 0:
            starts.append(position)
            offset = position + len(start_marker)
        for turn_index, position in enumerate(starts[1:], start=1):
            if not rendered[:position].endswith(previous_end):
                raise AssertionError(
                    f"{name}: turn {turn_index} starts with {start_marker!r} "
                    f"without preceding {previous_end!r}"
                )


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.data)
    kwargs_by_name = {name: json.loads(payload) for name, payload in args.template_kwargs}
    expected_generation_eos = {
        name: [int(token_id) for token_id in payload.split(",")]
        for name, payload in args.expected_generation_eos
    }

    for name, tokenizer_path in args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(tokenizer_path)
        actual_generation_eos = generation_config.eos_token_id
        actual_generation_eos = (
            list(actual_generation_eos) if isinstance(actual_generation_eos, (list, tuple)) else [actual_generation_eos]
        )
        if tokenizer.eos_token_id not in actual_generation_eos:
            raise AssertionError(
                f"{name}: tokenizer EOS {tokenizer.eos_token_id} is absent from generation EOS {actual_generation_eos}"
            )
        if name in expected_generation_eos and actual_generation_eos != expected_generation_eos[name]:
            raise AssertionError(
                f"{name}: expected generation EOS {expected_generation_eos[name]}, got {actual_generation_eos}"
            )
        template_kwargs = kwargs_by_name.get(name, {})
        config = {
            "pad_mode": "no_padding",
            "max_length": args.max_length,
            "truncation": "right",
            "apply_chat_template_kwargs": template_kwargs,
        }
        dataset = MultiTurnSFTDataset(args.data, tokenizer, config)
        lengths = []
        trained_tokens = []

        for index in range(len(dataset)):
            messages = frame.iloc[index]["messages"].tolist()
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **template_kwargs
            )
            assert_turn_boundaries(name, rendered)
            for message in messages:
                content = str(message["content"])
                if message["role"] == "assistant" and content not in rendered:
                    raise AssertionError(f"{name}: assistant content changed at row {index}")

            expected_open = sum(str(message["content"]).count("<think>") for message in messages)
            expected_close = sum(str(message["content"]).count("</think>") for message in messages)
            actual = (rendered.count("<think>"), rendered.count("</think>"))
            if actual != (expected_open, expected_close):
                raise AssertionError(
                    f"{name}: thinking tags changed at row {index}: "
                    f"expected={(expected_open, expected_close)} actual={actual}"
                )

            item = dataset[index]
            trained = int(item["loss_mask"].sum().item())
            if trained <= 0:
                raise AssertionError(f"{name}: zero trained tokens at row {index}")
            lengths.append(len(item["input_ids"]))
            trained_tokens.append(trained)

        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=SFTTensorCollator("no_padding"))))
        if not batch["input_ids"].is_nested or not batch["loss_mask"].is_nested:
            raise AssertionError(f"{name}: no-padding collator did not produce nested tensors")

        print(
            f"{name}: PASS rows={len(dataset)} length=[{min(lengths)},{max(lengths)}] "
            f"trained_tokens=[{min(trained_tokens)},{max(trained_tokens)}] "
            f"eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r} "
            f"generation_eos={actual_generation_eos}"
        )


if __name__ == "__main__":
    main()

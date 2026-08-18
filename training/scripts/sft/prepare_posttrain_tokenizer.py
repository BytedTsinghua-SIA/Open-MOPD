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

"""Attach an official post-training chat template to an unchanged Base vocabulary."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--template-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eos-token", required=True)
    parser.add_argument("--pad-token")
    parser.add_argument(
        "--generation-eos-token",
        action="append",
        default=[],
        help="Generation stop token. Repeat to preserve a model family's canonical multi-EOS list.",
    )
    parser.add_argument("--generation-bos-token")
    parser.add_argument("--generation-pad-token")
    parser.add_argument(
        "--smollm3-preserve-system-think-tags",
        action="store_true",
        help="Disable SmolLM3's /think substring rewrite, which corrupts literal </think> in system prompts.",
    )
    parser.add_argument(
        "--smollm3-fix-system-boundary",
        action="store_true",
        help="Ensure SmolLM3 closes its metadata/system turn even when no tools are supplied.",
    )
    parser.add_argument(
        "--qwen-preserve-assistant-content",
        action="store_true",
        help="Disable Qwen's inference-time reasoning extraction so every SFT assistant turn remains byte-for-byte intact.",
    )
    parser.add_argument("--fixed-date", help="Replace SmolLM3's runtime date with this reproducible date string.")
    return parser.parse_args()


def prepare_chat_template(template: str, args: argparse.Namespace) -> str:
    if args.qwen_preserve_assistant_content:
        old = '''    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}'''
        new = '''    {%- elif message.role == "assistant" %}
        {# OpenOPD data already contains the complete <think>...</think> block. #}
        {{- '<|im_start|>' + message.role + '\\n' + content }}'''
        if old not in template:
            raise ValueError("Qwen assistant-reasoning rewrite block was not found in the supplied template")
        template = template.replace(old, new, 1)

    if args.smollm3_preserve_system_think_tags:
        old = '''  {%- if "/no_think" in system_message -%}
    {%- set reasoning_mode = "/no_think" -%}
  {%- elif "/think" in system_message -%}
    {%- set reasoning_mode = "/think" -%}
  {%- endif -%}
  {%- set custom_instructions = system_message.replace("/no_think", "").replace("/think", "").rstrip() -%}'''
        new = '''  {# Preserve literal <think>...</think> instructions from OpenOPD data. #}
  {%- set custom_instructions = system_message.rstrip() -%}'''
        if old not in template:
            raise ValueError("SmolLM3 thinking-directive block was not found in the supplied template")
        template = template.replace(old, new, 1)

    if args.smollm3_fix_system_boundary:
        old = '''    {{- "\\n\\n" -}}
    {{- "<|im_end|>\\n" -}}
  {%- endif -%}
{%- endif -%}'''
        new = '''    {{- "\\n\\n" -}}
  {%- endif -%}
  {{- "<|im_end|>\\n" -}}
{%- endif -%}'''
        if old in template:
            template = template.replace(old, new, 1)
        elif '<|im_end|>\\n" -}}\n{%- endif -%}\n{#' not in template:
            raise ValueError("SmolLM3 system-turn boundary block was not found in the supplied template")

    if args.fixed_date is not None:
        old = '{%- set today = strftime_now("%d %B %Y") -%}'
        new = f'{{%- set today = "{args.fixed_date}" -%}}'
        if old not in template:
            raise ValueError("SmolLM3 runtime-date block was not found in the supplied template")
        template = template.replace(old, new, 1)

    return template


def token_id(tokenizer, token: str | None, *, field: str) -> int | None:
    if token is None:
        return None
    token_id_value = tokenizer.convert_tokens_to_ids(token)
    if token_id_value is None or token_id_value == tokenizer.unk_token_id:
        raise ValueError(f"{field} token is absent from Base vocabulary: {token!r}")
    return int(token_id_value)


def save_generation_config(tokenizer, args: argparse.Namespace, output: Path) -> GenerationConfig:
    generation_eos_tokens = args.generation_eos_token or [args.eos_token]
    generation_eos_ids = [token_id(tokenizer, token, field="generation EOS") for token in generation_eos_tokens]
    generation_eos_id: int | list[int]
    if len(generation_eos_ids) == 1:
        generation_eos_id = generation_eos_ids[0]
    else:
        generation_eos_id = generation_eos_ids

    generation_config = GenerationConfig(
        bos_token_id=token_id(tokenizer, args.generation_bos_token, field="generation BOS"),
        eos_token_id=generation_eos_id,
        pad_token_id=token_id(tokenizer, args.generation_pad_token, field="generation PAD"),
    )
    generation_config.save_pretrained(output)
    return generation_config


def main() -> None:
    args = parse_args()
    base = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    template = AutoTokenizer.from_pretrained(args.template_tokenizer, trust_remote_code=True)
    if not template.chat_template:
        raise ValueError(f"template tokenizer has no chat template: {args.template_tokenizer}")

    original_vocab = base.get_vocab()
    if args.eos_token not in original_vocab:
        raise ValueError(f"EOS token is absent from Base vocabulary: {args.eos_token!r}")
    if args.pad_token is not None and args.pad_token not in original_vocab:
        raise ValueError(f"PAD token is absent from Base vocabulary: {args.pad_token!r}")

    base.chat_template = prepare_chat_template(template.chat_template, args)
    base.eos_token = args.eos_token
    if args.pad_token is not None:
        base.pad_token = args.pad_token

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(output)
    generation_config = save_generation_config(base, args, output)

    reloaded = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    if reloaded.get_vocab() != original_vocab:
        raise RuntimeError("prepared tokenizer changed the Base token-to-id mapping")
    if not reloaded.chat_template:
        raise RuntimeError("prepared tokenizer lost its chat template after save/reload")
    print(
        f"prepared {output}: vocab={len(original_vocab)} eos={reloaded.eos_token!r} "
        f"pad={reloaded.pad_token!r} generation_eos={generation_config.eos_token_id!r}"
    )


if __name__ == "__main__":
    main()

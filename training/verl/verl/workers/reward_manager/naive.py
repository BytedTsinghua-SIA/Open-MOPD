# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# answer_length_cap (alc) default schedule: [offset_from_max_resp_len, answer_cap],
# sorted by offset DESCENDING. The answer reward (0..0.8) is capped to a value that
# linearly decays as the response gets longer; think_score is unaffected. At the
# budget (offset 0) the cap is 0, so a truncated answer (answer_score already 0)
# stays 0 and is graded down rather than masked out.
_ALC_DEFAULT_BREAKPOINTS = [[12288, 0.8], [8192, 0.65], [4096, 0.4], [0, 0.0]]


def _cfg_get(cfg, key, default):
    """Read a key from a dict / OmegaConf DictConfig / attr-style namespace."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _answer_length_cap(resp_len, max_resp_len, breakpoints):
    """Piecewise-linear cap on the answer reward as a function of response length.

    ``breakpoints`` is a list of ``[offset_from_max, cap]`` sorted by offset
    descending (e.g. ``[[12288, 0.8], [8192, 0.65], [4096, 0.4], [0, 0.0]]``),
    where ``offset = max_resp_len - resp_len``. Responses shorter than the first
    breakpoint get its cap (no length penalty); the cap interpolates linearly
    between consecutive breakpoints and equals the last cap at/after the budget.
    """
    offset = float(max_resp_len) - float(int(resp_len))
    pts = [(float(o), float(c)) for o, c in breakpoints]
    if offset >= pts[0][0]:
        return pts[0][1]
    for (o_hi, c_hi), (o_lo, c_lo) in zip(pts, pts[1:]):
        if o_lo <= offset <= o_hi:
            if o_hi == o_lo:
                return c_lo
            frac = (offset - o_lo) / (o_hi - o_lo)
            return c_lo + frac * (c_hi - c_lo)
    return pts[-1][1]


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        enable_format_reward=False,
        max_workers=1,
        overlong_buffer_cfg=None,
        max_resp_len=None,
        length_cap_cfg=None,
        compute_true_reward=True,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            enable_format_reward: Whether to enable format reward. Defaults to False.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.enable_format_reward = enable_format_reward
        self.max_workers = max(1, int(max_workers))
        # DAPO soft overlong reward shaping (optional). When enabled, responses whose
        # length exceeds (max_resp_len - overlong_buffer.len) receive a linearly growing
        # penalty, reaching -penalty_factor at exactly max_resp_len. See DAPO §3.4.
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        # answer_length_cap (alc): optional length->answer-reward cap schedule. Only
        # active when length_cap_cfg.enable is set, so other runs are unaffected.
        self.length_cap_cfg = length_cap_cfg
        self.compute_true_reward = bool(compute_true_reward)
        if self.overlong_buffer_cfg is not None and getattr(self.overlong_buffer_cfg, "enable", False):
            assert self.max_resp_len is not None, (
                "max_resp_len must be provided when overlong_buffer_cfg.enable=True"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be >= overlong_buffer_cfg.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # OPD's learning signal has already been materialized as dense teacher-KL
        # rm_scores. Rule scoring here is diagnostic only and, for code, can cost
        # thousands of sandbox jobs after the GPU teacher pass has finished. Recipes
        # may disable that diagnostic without changing the tensor used for learning.
        if "rm_scores" in data.batch.keys() and not self.compute_true_reward:
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        format_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32) # (batch_size, )
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        def process_one(i: int) -> dict[str, Any]:
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.enable_format_reward:
                if r"\boxed" in response_str:
                    format_score = 1.0
                else:
                    format_score = 0.0
            else:
                format_score = None

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
            else:
                reward = score

            # Length shaping. Two mutually exclusive schemes:
            #  (1) answer_length_cap (alc): when length_cap_cfg.enable, cap the *answer*
            #      reward by a piecewise-linear schedule of response length (think_score
            #      kept intact). Replaces the soft overlong penalty AND removes any
            #      truncation amnesty -- overlong/truncated correct answers are graded
            #      down to 0, so they are still suppressed by GRPO advantage.
            #  (2) DAPO soft overlong penalty: graded -penalty_factor as length approaches
            #      the budget. Original default; untouched when alc is off.
            overlong_reward = 0.0
            length_cap = None
            if (
                self.length_cap_cfg is not None
                and _cfg_get(self.length_cap_cfg, "enable", False)
                and isinstance(score, dict)
            ):
                breakpoints = _cfg_get(self.length_cap_cfg, "breakpoints", _ALC_DEFAULT_BREAKPOINTS)
                answer_score = float(score.get("answer_score", 0.0))
                think_score = float(score.get("think_score", 0.0))
                length_cap = _answer_length_cap(valid_response_length, self.max_resp_len, breakpoints)
                reward = think_score + min(answer_score, length_cap)
            elif self.overlong_buffer_cfg is not None and getattr(self.overlong_buffer_cfg, "enable", False):
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = int(valid_response_length) - expected_len
                penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0.0)
                reward = reward + overlong_reward

            return {
                "index": i,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "score": score,
                "reward": reward,
                "overlong_reward": overlong_reward,
                "length_cap": length_cap,
                "format_score": format_score,
                "valid_response_length": int(valid_response_length.item()),
            }

        if self.max_workers == 1 or len(data) <= 1:
            results = [process_one(i) for i in range(len(data))]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_one, range(len(data))))

        for result in results:
            i = result["index"]
            data_source = result["data_source"]
            score = result["score"]
            reward_tensor[i, result["valid_response_length"] - 1] = result["reward"]
            if self.overlong_buffer_cfg is not None and getattr(self.overlong_buffer_cfg, "enable", False):
                reward_extra_info["overlong_reward"].append(result["overlong_reward"])
                reward_extra_info["overlong"].append(result["overlong_reward"] < 0)
            if result.get("length_cap") is not None:
                reward_extra_info["length_cap"].append(result["length_cap"])
                reward_extra_info["length_capped"].append(result["length_cap"] < 0.8)
            if result["format_score"] is not None:
                format_tensor[i] = result["format_score"]
            if isinstance(score, dict):
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", result["prompt_str"])
                print("[response]", result["response_str"])
                print("[ground_truth]", result["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # Caculate the reward using reward_fn first, we want to know true reward scores, but we still use rm_scores for training
        if "rm_scores" in data.batch.keys():
            print(f"Now we are using rm_scores!")
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                reward_extra_info["true_reward_score"] = reward_tensor
                if self.enable_format_reward:
                    print("Format mask has been added to reward_extra_info!")
                    reward_extra_info["format_mask"] = format_tensor
                print("True reward score has been added to reward_extra_info!")
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

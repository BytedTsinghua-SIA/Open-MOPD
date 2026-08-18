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

import pytest
import torch

from verl.workers.actor.dp_actor import _align_rmpad_topk_ids_for_ulysses


@pytest.mark.parametrize(
    ("sp_rank", "expected_rows"),
    [
        (0, [[0, 100], [1, 101], [2, 102]]),
        (1, [[3, 103], [4, 104], [0, 0]]),
    ],
)
def test_align_global_rmpad_topk_ids_like_ulysses_input_slice(sp_rank, expected_rows):
    # Five packed tokens are padded to six, then split into contiguous chunks
    # of three just like ulysses_pad_and_slice_inputs(..., sp_size=2).
    topk_ids = torch.tensor([[i, i + 100] for i in range(5)])

    local_ids = _align_rmpad_topk_ids_for_ulysses(
        topk_ids,
        local_token_count=3,
        padding_size=1,
        sp_size=2,
        sp_rank=sp_rank,
    )

    assert local_ids.tolist() == expected_rows
    assert local_ids.is_contiguous()


def test_align_accepts_ids_already_sliced_by_upstream():
    local_ids = torch.tensor([[7, 8], [9, 10], [11, 12]])

    aligned = _align_rmpad_topk_ids_for_ulysses(
        local_ids,
        local_token_count=3,
        padding_size=1,
        sp_size=2,
        sp_rank=1,
    )

    assert aligned.data_ptr() == local_ids.data_ptr()


@pytest.mark.parametrize(
    ("topk_ids", "kwargs", "match"),
    [
        (torch.zeros(1, 2, 3), {}, "must be \\[tokens, K\\]"),
        (torch.zeros(4, 2), {}, "do not align"),
        (torch.zeros(5, 2), {"sp_rank": 2}, "invalid Ulysses SP rank"),
    ],
)
def test_align_rejects_ambiguous_or_invalid_shapes(topk_ids, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _align_rmpad_topk_ids_for_ulysses(
            topk_ids,
            local_token_count=3,
            padding_size=1,
            sp_size=2,
            sp_rank=kwargs.get("sp_rank", 0),
        )

from __future__ import annotations

import os
from unittest import mock

import torch

from verl.utils.distributed import initialize_global_process_group


def test_initialize_global_process_group_passes_device_id(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "35")
    monkeypatch.setenv("WORLD_SIZE", "64")

    with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
        with mock.patch.object(torch.distributed, "init_process_group") as init_pg:
            with mock.patch("verl.utils.distributed.get_torch_device") as get_torch_device:
                cuda = mock.Mock()
                get_torch_device.return_value = cuda

                local_rank, rank, world_size = initialize_global_process_group(timeout_second=60)

    assert (local_rank, rank, world_size) == (3, 35, 64)
    cuda.set_device.assert_called_once_with(3)
    _, kwargs = init_pg.call_args
    assert kwargs["backend"] == "nccl"
    assert kwargs["device_id"] == torch.device("cuda", 3)

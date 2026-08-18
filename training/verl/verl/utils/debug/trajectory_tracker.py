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
"""
Trajectory tracker can be inserted into code to save intermediate results to
a shared local directory. Each process first moves tensors to CPU.
"""

import io
import os
import tempfile
from collections import deque

import ray
import torch

from verl.utils.local_io import copy, makedirs

remote_copy = ray.remote(copy)


@ray.remote
def save_local(data: io.BytesIO, name, output_dir, verbose):
    filename = name + ".pth"
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_filepath = os.path.join(tmpdirname, filename)
        with open(local_filepath, "wb") as f:
            f.write(data.getbuffer())
        if verbose:
            print(f"Saving {local_filepath} to {output_dir}")
        try:
            copy(local_filepath, output_dir)
        except Exception as e:
            print(e)


@ray.remote
class TrajectoryTracker:
    def __init__(self, output_dir, verbose) -> None:
        self.output_dir = output_dir
        makedirs(output_dir, exist_ok=True)
        self.verbose = verbose

        self.handle = deque()

    def dump(self, data: io.BytesIO, name):
        # get a temp file and write to it
        self.handle.append(save_local.remote(data, name, self.output_dir, self.verbose))

    def wait_for_writes(self):
        while len(self.handle) != 0:
            future = self.handle.popleft()
            ray.get(future)


def dump_data(data, name):
    enable = os.getenv("VERL_ENABLE_TRACKER", "0") == "1"
    if not enable:
        return
    buffer = io.BytesIO()
    torch.save(data, buffer)
    tracker = get_trajectory_tracker()
    ray.get(tracker.dump.remote(buffer, name))


def get_trajectory_tracker():
    output_dir = os.getenv("VERL_TRACKER_OUTPUT_DIR", default=None)
    verbose = os.getenv("VERL_TRACKER_VERBOSE", default="0") == "1"
    assert output_dir is not None
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(
        output_dir, verbose
    )
    return tracker


if __name__ == "__main__":
    # testing
    os.environ["VERL_ENABLE_TRACKER"] = "1"
    os.environ["VERL_TRACKER_OUTPUT_DIR"] = "~/debug/test"

    @ray.remote
    def process(iter):
        data = {"obs": torch.randn(10, 20)}
        dump_data(data, f"process_{iter}_obs")

    ray.init()

    output_lst = []

    for i in range(10):
        output_lst.append(process.remote(i))

    out = ray.get(output_lst)

    tracker = get_trajectory_tracker()
    ray.get(tracker.wait_for_writes.remote())

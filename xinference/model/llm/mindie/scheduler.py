# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
import threading
from concurrent.futures import Future as ConcurrentFuture
from queue import Queue
from typing import Optional

import torch

from ....core.scheduler import (
    XINFERENCE_NON_STREAMING_ABORT_FLAG,
    XINFERENCE_STREAMING_ABORT_FLAG,
    XINFERENCE_STREAMING_DONE_FLAG,
    XINFERENCE_STREAMING_ERROR_FLAG,
)
from ....core.scheduler import InferenceRequest as _InferenceRequest
from ....core.scheduler import SchedulerActor as _SchedulerActor


class InferenceRequest(_InferenceRequest):
    need_blocks: int
    need_slots: int
    block_tables: Optional[torch.Tensor]
    slot_tables: Optional[torch.Tensor]

    def __init__(self, prompt, future_or_queue, is_prefill, *args, **kwargs):
        super().__init__(prompt, future_or_queue, is_prefill, *args, **kwargs)

        self._input_ids = None
        self.input_length = None
        self.max_out_length = None
        self.need_blocks = 0
        self.need_slots = 0
        self.block_tables = None
        self.slot_tables = None
        self.out_token_list = []

    def __repr__(self):
        return f"InstanceRequest (request_id={self.request_id})"

    def __getstate__(self):
        d = self.__dict__.copy()
        fq = d.pop("future_or_queue")
        if isinstance(fq, asyncio.Queue):
            d["is_future"] = True
        else:
            d["is_future"] = False
        return d

    def __setstate__(self, state):
        is_future = state.pop("is_future")
        if is_future:
            state["future_or_queue"] = asyncio.Queue()
        else:
            state["future_or_queue"] = ConcurrentFuture()
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, new_input_ids: torch.Tensor):
        self._input_ids = new_input_ids.flatten()
        self.input_length = self._input_ids.numel()

    @property
    def ignore_eos(self):
        return bool(self.generate_config.get("ignore_eos", True))

    @property
    def req_id(self):
        return self.request_id

    def calc_blocks(self, block_size: int):
        assert self.input_ids is not None

        self.need_blocks = math.ceil(
            (self.input_length + self.max_out_length) / block_size
        )
        self.need_slots = self.need_blocks * block_size


class SchedulerActor(_SchedulerActor):
    _batch_inference_q: Optional[Queue]

    def __init__(self, rank: int, max_num_seqs: int):
        super().__init__()
        self.rank = rank
        self._batch_inference_q = None
        self._max_num_seqs = max_num_seqs

    @classmethod
    def uid(cls):
        return "MindIEScheduler"

    def set_batch_inference_q(self, q: Queue):
        self._batch_inference_q = q

    def get_max_num_seqs(self):
        return self._max_num_seqs

    async def add_request(self, prompt: str, future_or_queue, *args, **kwargs):
        req = InferenceRequest(prompt, future_or_queue, True, *args, **kwargs)
        rid = req.request_id
        if rid is not None:
            if rid in self._id_to_req:
                raise KeyError(f"Request id: {rid} has already existed!")
            self._id_to_req[rid] = req
        self._waiting_queue.append(req)

    async def step(self):
        req_list = self._handle_request()
        if not req_list:
            return

        done_event = threading.Event()
        self._batch_inference_q.put((req_list, done_event))
        # wait for done
        await asyncio.to_thread(done_event.wait)

        stopped_batch_indexes = set()
        for idx, r in enumerate(req_list):
            if r.stream:
                for completion in r.completion:
                    await r.future_or_queue.put(completion)
                r.completion = []

            if not r.stopped:
                self._running_queue.append(r)
            else:
                if r.new_tokens:
                    stopped_batch_indexes.add(idx)
                rid = r.request_id
                # clear data structure
                if rid is not None:
                    self._id_to_req.pop(rid, None)
                    self._abort_req_ids.discard(rid)

                if r.aborted:  # stop due to abort
                    # handle abort result
                    if r.stream:
                        await r.future_or_queue.put(XINFERENCE_STREAMING_ABORT_FLAG)
                    else:
                        r.future_or_queue.set_result(
                            XINFERENCE_NON_STREAMING_ABORT_FLAG
                        )
                else:
                    if r.error_msg is None:  # normal stop
                        if not r.stream:
                            r.future_or_queue.set_result(r.completion[0])
                        else:
                            await r.future_or_queue.put(XINFERENCE_STREAMING_DONE_FLAG)
                    # Abnormal stop, currently indicates that the parameter check does not pass,
                    # and does not participate in the inference
                    else:
                        if not r.stream:
                            r.future_or_queue.set_exception(ValueError(r.error_msg))
                        else:
                            await r.future_or_queue.put(
                                XINFERENCE_STREAMING_ERROR_FLAG + r.error_msg
                            )

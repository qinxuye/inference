# Copyright 2022-2025 XProbe Inc.
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
from typing import Dict, Optional

import mlx.core as mx
import xoscar as xo


class ReceiverActor(xo.StatelessActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._receiving_fut = asyncio.Future()
        self._set_result_fut = asyncio.Future()

    @classmethod
    def gen_uid(cls, uid: str, rank: int):
        return f"Receiver-{uid}-{rank}"

    def send(self, data: mx.array):
        if self._receiving_fut.done():
            # new iteration
            self._receiving_fut = asyncio.Future()
        self._receiving_fut.set_result(data)

    async def recv(self):
        return await self._receiving_fut

    def set_result(self, result: mx.array):
        if self._set_result_fut.done():
            # new iteration
            self._set_result_fut = asyncio.Future()
        self._set_result_fut.set_result(result)

    async def get_result(self):
        return await self._set_result_fut


class DistributedModelMixin:
    rank: int
    world_size: int
    model_uid: Optional[str]
    address: Optional[str]
    _receiver_ref: Optional[xo.ActorRef[ReceiverActor]]
    rank_to_addresses: Optional[Dict[int, str]]

    layers: list

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.model_uid = None
        self.loop = None
        self.address = None
        # actor ref
        self._receiver_ref = None
        self.rank_to_addresses = None

    def prepare(self):
        coro = xo.create_actor(
            ReceiverActor,
            uid=ReceiverActor.gen_uid(self.model_uid, self.rank),
            address=self.address,
        )
        self._receiver_ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _send_stage_result(self, result: mx.array):
        assert self.rank > 0
        assert self.rank_to_addresses is not None
        assert self.model_uid is not None
        last_rank = self.rank - 1
        coro = xo.actor_ref(
            uid=ReceiverActor.gen_uid(self.model_uid, last_rank),
            address=self.rank_to_addresses[last_rank],
        )
        receiver_ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        coro = receiver_ref.send(result)
        asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _wait_prev_stage_result(self):
        coro = self._receiver_ref.recv()
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _get_result(self):
        coro = xo.actor_ref(
            uid=ReceiverActor.gen_uid(self.model_uid, 0),
            address=self.rank_to_addresses[0],
        )
        receiver_ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        coro = receiver_ref.get_result()
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def _set_result(self, result: mx.array):
        assert self.model_uid is not None
        assert self.rank_to_addresses is not None
        coro = xo.actor_ref(
            uid=ReceiverActor.gen_uid(self.model_uid, 0),
            address=self.rank_to_addresses[0],
        )
        receiver_ref = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        coro = receiver_ref.set_result(result)
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def pipeline(self):
        pipeline_size, rank = self.world_size, self.rank
        layers_per_rank = len(self.layers) // pipeline_size
        extra = len(self.layers) - layers_per_rank * pipeline_size
        if self.rank < extra:
            layers_per_rank += 1
        self.start_idx = (pipeline_size - rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

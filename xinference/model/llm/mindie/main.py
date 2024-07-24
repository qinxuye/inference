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

import argparse
import asyncio
import os
from concurrent.futures import Future as ConcurrentFuture
from queue import Queue
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

import torch.distributed
import xoscar as xo
from atb_llm.utils.log import logger, print_log

from ....isolation import Isolation
from ....types import Completion, CompletionChunk
from .scheduler import InferenceRequest, SchedulerActor
from .stream_runner import StreamPARunner


def start_serve(**kwargs):
    rank = kwargs["rank"]
    local_rank = kwargs["local_rank"]
    world_size = kwargs["world_size"]

    if world_size > 1:
        torch.distributed.init_process_group()

    if rank == 0:
        to_send_kwargs = kwargs.copy()
        to_send_kwargs.pop("rank")
        to_send_kwargs.pop("local_rank")
        to_send_kwargs.pop("world_size")
        objects = [to_send_kwargs] * world_size
    else:
        objects = []
    if world_size > 1:
        output_list: List[dict] = [{}]  # type: ignore
        torch.distributed.scatter_object_list(output_list, objects)
        if rank > 0:
            kwargs = output_list[0]
            kwargs["rank"] = rank
            kwargs["local_rank"] = local_rank
            kwargs["world_size"] = world_size

    # launch model
    print_log(rank, logger.info, f"launching runner with: {kwargs}")
    pa_runner = StreamPARunner(**kwargs)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()
    return pa_runner


def serve(pa_runner: StreamPARunner, q: Queue):
    pa_runner.init_cache_manager()
    try:
        req_id_to_req: Dict[str, InferenceRequest] = dict()
        while True:
            req_list, done_event = None, None
            if rank == 0:
                req_list, done_event = q.get()
                objects = [req_list] * world_size
            else:
                objects = []
            if world_size > 1:
                output_objects: List[list] = [[]]
                torch.distributed.scatter_object_list(output_objects, objects)
                if rank > 0:
                    req_list = output_objects[0]
                    for r in req_list:
                        if r.request_id in req_id_to_req:
                            for k, v in r.__dict__.items():
                                setattr(req_id_to_req[r.request_id], k, v)
            req_list = [req_id_to_req.get(r.request_id, r) for r in req_list]  # type: ignore

            # batch inference
            pa_runner.batch_inference(req_list)
            if rank == 0:
                # set done event
                done_event.set()  # type: ignore

            # update req_id_to_req
            for req in req_list:
                if req.stopped:
                    req_id_to_req.pop(req.request_id, None)
                    continue
                if req.request_id not in req_id_to_req:
                    req_id_to_req[req.request_id] = req
    finally:
        pa_runner.empty_cache_manager()


class MindIEModelActor(xo.StatelessActor):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self._rank = kwargs["rank"]
        self._max_batch_size = kwargs["max_batch_size"]
        self._scheduler_ref = None
        self._kwargs_updated = asyncio.Event()
        self._is_ready = asyncio.Event()

    async def __post_create__(self):
        self._scheduler_ref = await xo.create_actor(
            SchedulerActor,
            self._rank,
            self._max_batch_size,
            address=self.address,
            uid=SchedulerActor.uid(),
        )

    async def wait_ready(self):
        return await self._is_ready.wait()

    def mark_ready(self):
        self._is_ready.set()

    def update_kwargs(self, **kwargs):
        self._kwargs.update(kwargs)
        self._kwargs_updated.set()

    async def wait_for_kwargs(self):
        await self._kwargs_updated.wait()

    def get_kwargs(self):
        return self._kwargs

    async def set_batch_inference_q(self, q: Queue):
        await self._scheduler_ref.set_batch_inference_q(q)

    @classmethod
    def uid(cls) -> str:
        return "MindIEModel"

    async def _queue_consumer(
        self, queue: asyncio.Queue, timeout: Optional[float] = None
    ) -> AsyncIterator[Any]:
        from .scheduler import (
            XINFERENCE_STREAMING_ABORT_FLAG,
            XINFERENCE_STREAMING_DONE_FLAG,
            XINFERENCE_STREAMING_ERROR_FLAG,
        )

        while True:
            # TODO: timeout setting
            res = await asyncio.wait_for(queue.get(), timeout)
            if res == XINFERENCE_STREAMING_DONE_FLAG:
                break
            elif res == XINFERENCE_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            elif isinstance(res, str) and res.startswith(
                XINFERENCE_STREAMING_ERROR_FLAG
            ):
                raise RuntimeError(res[len(XINFERENCE_STREAMING_ERROR_FLAG) :])
            else:
                yield res

    @xo.generator
    async def chat(
        self,
        prompt: str,
        *args,
        **kwargs,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        sanitized_generate_config: dict = args[-1]  # type: ignore
        stream = sanitized_generate_config.get("stream", False)

        if stream:
            assert self._scheduler_ref is not None
            queue: asyncio.Queue[Any] = asyncio.Queue()
            ret = self._queue_consumer(queue)
            await self._scheduler_ref.add_request(prompt, queue, *args, **kwargs)
            return ret
        else:
            from .scheduler import XINFERENCE_NON_STREAMING_ABORT_FLAG

            future: ConcurrentFuture = ConcurrentFuture()
            await self._scheduler_ref.add_request(prompt, future, *args, **kwargs)  # type: ignore
            fut: asyncio.Future = asyncio.wrap_future(future)
            result = await fut
            if result == XINFERENCE_NON_STREAMING_ABORT_FLAG:
                raise RuntimeError(
                    f"This request has been cancelled by another `abort_request` request."
                )
            return result


async def start_actor_pool(**kwargs):
    port = kwargs["port"]
    address = f"127.0.0.1:{port}"
    server = await xo.create_actor_pool(
        address,
        n_process=0,
        auto_recover="process",
        subprocess_start_method="forkserver",
    )

    model_ref = await xo.create_actor(
        MindIEModelActor,
        address=server.external_address,
        uid=MindIEModelActor.uid(),
        **kwargs,
    )

    await model_ref.wait_for_kwargs()
    return model_ref, await model_ref.get_kwargs()


async def set_status(model_ref: xo.ActorRefType[MindIEModelActor], q: Queue):
    await model_ref.set_batch_inference_q(q)
    await model_ref.mark_ready()


def main(**kwargs):
    rank = kwargs["rank"]

    q = Queue()
    isolation = model_ref = None
    if rank == 0:
        isolation = Isolation(asyncio.new_event_loop())
        isolation.start()
        model_ref, kwargs = isolation.call(start_actor_pool(**kwargs))

    runner = start_serve(**kwargs)

    if rank == 0:
        isolation.call(set_status(model_ref, q))

    serve(runner, q)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="xoscar port to communicate with")
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--max_prefill_tokens", type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=15)
    parser.add_argument("--block_size", type=int, default=128)

    parser.add_argument("--is_flash_model", action="store_false")

    parser.add_argument(
        "--num_beams", type=int, help="Use beam search if num_beams >1", default=1
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--use_refactor", type=bool, default=True)
    parser.add_argument("--ignore_eos", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    input_dict = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        **vars(args),
    }

    main(**input_dict)

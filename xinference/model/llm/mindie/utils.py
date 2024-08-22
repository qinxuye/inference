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

import os
import time
import weakref
from typing import Dict, List, Tuple

import torch
from atb_llm.utils.log import logger, print_log

from ....core.model import OutOfMemoryError
from ....types import max_tokens_field
from ..transformers.utils import (
    _get_completion,
    _get_completion_chunk,
    prepare_logits_processor,
)
from .scheduler import InferenceRequest
from .server.batch import Batch
from .server.cache import CacheManager
from .server.generate import generate_token


def _get_token_from_logits(
    req: InferenceRequest, logits, temperature, repetition_penalty, top_p, top_k
):
    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    if logits_processor:
        if repetition_penalty > 1.0:
            tmp_output_ids = torch.as_tensor(
                [req.prompt_tokens + req.new_tokens], device=logits.device
            )
        else:
            tmp_output_ids = None
        last_token_logits = logits_processor(tmp_output_ids, logits.reshape(1, -1))[0]
    else:
        last_token_logits = logits

    if temperature < 1e-5 or top_p < 1e-8:  # greedy
        _, indices = torch.topk(last_token_logits, 2)
    else:
        probs = torch.softmax(last_token_logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=2)
    token = indices[0].int().item()
    return token


@torch.inference_mode()
def _batch_inference_one_step_internal(
    req_list: List[InferenceRequest],
    model_uid,
    model,
    tokenizer,
    context_len: int,
    rank: int,
    world_size: int,
    block_size: int,
    cache_manager: CacheManager,
    req_to_batches: weakref.WeakKeyDictionary,
    decode_round: int = 16,
    bos_flag: str = "<bos_stream>",
    eos_flag: str = "<eos_stream>",
):
    # need to judge stopped here,
    # since some requests state may change to stopped due to invalid parameters, e.g. max_src_len
    valid_req_list = [r for r in req_list if not r.stopped]
    if not valid_req_list:
        return
    generate_config_mapping: Dict[InferenceRequest, Tuple] = {
        r: r.get_generate_configs(tokenizer.eos_token_id) for r in valid_req_list
    }
    s_time = time.time()

    free_block = cache_manager.get_free_block_num()
    prefill_reqs = []
    prompts = []
    decode_reqs = []
    for r in valid_req_list:
        if r.is_prefill:
            prompts.append(r.full_prompt)
            prefill_reqs.append(r)
        else:
            decode_reqs.append(r)

    if prompts:  # prefill first
        total_need_blocks = 0
        input_ids: List[List[int]] = tokenizer(prompts, padding=False).input_ids
        max_out_length = []
        get_next_token_methods = []
        for i, input_id in enumerate(input_ids):
            req = prefill_reqs[i]
            max_new_tokens = int(
                req.sanitized_generate_config.get(
                    "max_output_length", max_tokens_field.default
                )
            )
            max_out_length.append(max_new_tokens)
            req.max_out_length = max_new_tokens
            max_src_len = context_len - max_new_tokens - 8
            req.prompt_tokens = input_id[-max_src_len:]
            req.input_ids = torch.tensor(req.prompt_tokens, dtype=torch.int64)
            req.calc_blocks(block_size)
            cur_need_blocks = req.need_blocks
            if total_need_blocks + cur_need_blocks > free_block:
                raise OutOfMemoryError(
                    f"req: {req.request_id} out of memory, need block:"
                    + f"{total_need_blocks + cur_need_blocks} is more than free block {free_block}"
                )
            total_need_blocks += cur_need_blocks

            (
                max_new_tokens,
                stream_interval,
                include_usage,
                stop_str,
                stop_token_ids,
                temperature,
                repetition_penalty,
                top_p,
                top_k,
            ) = generate_config_mapping[req]
            get_next_token_method = lambda _logits: _get_token_from_logits(
                req, _logits, temperature, repetition_penalty, top_p, top_k
            )
            get_next_token_methods.append(get_next_token_method)

        batch = Batch(prefill_reqs)  # type: ignore
        for prefill_req in prefill_reqs:
            req_to_batches[prefill_req] = batch
        cache_manager.allocate(batch)
        generate_token(
            model,
            cache_manager,
            batch,
            rank,
            get_next_token_methods=get_next_token_methods,
        )

        for req, server_req in zip(prefill_reqs, batch.req_list):
            req.is_prefill = False
            req.append_new_token(server_req.out_token_list[-1])

    # decode
    batches = list({req_to_batches[r] for r in valid_req_list})
    if len(batches) > 1:
        Batch.concatenate(batches)
        for r in valid_req_list:
            req_to_batches[r] = batches[0]
    get_next_token_methods = []
    max_out_length = []
    stop_token_mapping: Dict[InferenceRequest, int] = {}
    output_mapping: Dict[InferenceRequest, str] = {}
    # only decode phase, run rounds
    for _i in range(decode_round):
        for i, r in enumerate(batches[0].req_list):
            (
                max_new_tokens,
                stream_interval,
                include_usage,
                stop_str,
                stop_token_ids,
                temperature,
                repetition_penalty,
                top_p,
                top_k,
            ) = generate_config_mapping[r]
            get_next_token_method = lambda _logits: _get_token_from_logits(
                r, _logits, temperature, repetition_penalty, top_p, top_k
            )
            get_next_token_methods.append(get_next_token_method)
            max_new_tokens = int(
                r.sanitized_generate_config.get(
                    "max_output_length", max_tokens_field.default
                )
            )
            max_out_length.append(max_new_tokens)

        generate_token(
            model,
            cache_manager,
            batches[0],
            rank,
            get_next_token_methods=get_next_token_methods,
        )

        for i, r in enumerate(valid_req_list):
            (
                max_new_tokens,
                stream_interval,
                include_usage,
                stop_str,
                stop_token_ids,
                temperature,
                repetition_penalty,
                top_p,
                top_k,
            ) = generate_config_mapping[r]
            token = r.out_token_list[-1]
            r.append_new_token(token)

            output = None
            if not r.stopped:
                stopped = token in stop_token_ids

                if stopped:
                    finish_reason = "stop"
                elif len(r.new_tokens) >= max_out_length[i]:
                    finish_reason = "length"
                    stopped = True
                else:
                    finish_reason = None

                # handle stop str
                if stop_str and r not in output_mapping:
                    output = tokenizer.decode(
                        r.new_tokens,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    if isinstance(stop_str, str):
                        stop_str = [stop_str]
                    for stop in stop_str:
                        pos = output.rfind(stop)
                        if pos != -1:
                            output = output[:pos]
                            output_mapping[r] = output
                            stopped = True
                            finish_reason = "stop"
                            break

                r.stopped = stopped
                r.finish_reason = finish_reason

            if r.stopped and r not in stop_token_mapping and r not in output_mapping:
                stop_token_mapping[r] = _i + 1

            if r.stream:
                """
                Note that you can't just decode based on the newest r.new_tokens here,
                which may destroy the integrity of the parsed characters,
                and at the same time is not good at handling some special characters.
                So the implementation here is to decode all the tokens that have been generated each time,
                and then take the slice.
                """
                if r.stopped or len(r.new_tokens) % stream_interval == 0:
                    if output is None:
                        output = tokenizer.decode(
                            r.new_tokens,
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )

                    if r.last_output_length == 0:
                        r.completion.append(bos_flag)

                    # this special character is mainly for qwen
                    output = output.strip("�")
                    output = output[r.last_output_length :]
                    r.last_output_length += len(output)

                    completion_chunk = _get_completion_chunk(
                        output, r.chunk_id, r.finish_reason, model_uid, r, False
                    )
                    r.completion.append(completion_chunk)
                    if r.stopped:
                        r.completion.append(eos_flag)

                    # last round, handle stream result
                    # append usage information when enable `include_usage` for OPENAI API compatibility
                    # The reason for counting the usage in the last round of the iteration is that,
                    # these tokens are real generated and should be counted.
                    if r.stopped and include_usage:
                        r.completion.append(
                            _get_completion_chunk(
                                "", r.chunk_id, r.finish_reason, model_uid, r, True
                            )
                        )
            else:
                # last round, handle non-stream result
                if r.stopped and _i == decode_round - 1:
                    invalid_token_num = decode_round - stop_token_mapping[r]
                    outputs = (
                        tokenizer.decode(
                            r.new_tokens[: -(invalid_token_num + 1)]
                            if r.finish_reason == "stop"
                            else r.new_tokens[:-invalid_token_num],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                        if r not in output_mapping
                        else output_mapping[r]
                    )
                    completion = _get_completion(
                        outputs, r.chunk_id, r.finish_reason, model_uid, r
                    )
                    r.completion = [completion]

    # synchronize stop status
    if rank == 0:
        stops = [r.stopped for r in valid_req_list]
        objects = [stops] * world_size
    else:
        objects = []
    if world_size > 1:
        output_objects: List[list] = [[]]
        torch.distributed.scatter_object_list(output_objects, objects)
        if rank > 0:
            stops = output_objects[0]
            for r, stop in zip(valid_req_list, stops):
                r.stopped = stop

    # filter the finished requests
    batches[0].filter(cache_manager)

    e_time = time.time()
    print_log(
        rank,
        logger.debug,
        f"Average throughput for a step: {(len(valid_req_list) + len(prompts)) / (e_time - s_time)} token/s.",
    )


def batch_inference_one_step(
    req_list: List[InferenceRequest],
    model_uid,
    model,
    tokenizer,
    context_len: int,
    rank: int,
    world_size: int,
    block_size: int,
    cache_manager: CacheManager,
    req_to_batches: weakref.WeakKeyDictionary,
    bos_flag: str = "<bos_stream>",
    eos_flag: str = "<eos_stream>",
):
    try:
        _batch_inference_one_step_internal(
            req_list,
            model_uid,
            model,
            tokenizer,
            context_len,
            rank,
            world_size,
            block_size,
            cache_manager,
            req_to_batches,
            bos_flag=bos_flag,
            eos_flag=eos_flag,
        )
    except OutOfMemoryError:
        logger.exception(
            f"Batch inference out of memory. "
            f"Xinference will restart the model: {model_uid}. "
            f"Please be patient for a few moments."
        )
        # Just kill the process and let xinference auto-recover the model
        os._exit(1)
    except Exception as e:
        logger.exception(f"Internal error for batch inference: {e}.")
        # TODO: handle this
        raise

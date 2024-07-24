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
import copy
import math
import os
import time

import torch
import torch_npu
from atb_llm.runner import ModelRunner
from atb_llm.utils.cpu_binding import NpuHbmInfo
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log

from .cache import CacheConfig, CacheManager, ModelConfig
from .generate import decode_token, generate_req
from .request import request_from_text, request_from_token


class PARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get("rank", "0")
        self.local_rank = kwargs.get("local_rank", "0")
        self.world_size = kwargs.get("world_size", "1")

        self.model_path = kwargs.get("model_path", None)
        self.input_text = kwargs.get("input_text", None)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", None)
        self.max_input_length = kwargs.get("max_input_length", None)
        self.max_prefill_tokens = kwargs.get("max_prefill_tokens", None)
        self.max_output_length = kwargs.get("max_output_length", None)
        self.is_flash_model = kwargs.get("is_flash_model", None)
        self.max_batch_size = kwargs.get("max_batch_size", None)
        self.use_refactor = kwargs.get("use_refactor", True)

        self.block_size = kwargs.get("block_size", None)

        self.model = ModelRunner(
            self.model_path,
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            max_position_embeddings=self.max_position_embeddings,
            # use_refactor=self.use_refactor,
        )
        self.tokenizer = self.model.tokenizer
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.model.load_weights()

        self.device = self.model.device
        self.model_config = ModelConfig(
            self.model.num_heads,
            self.model.num_kv_heads,
            self.model.head_size,
            self.model.num_layers,
            self.model.device,
            self.model.dtype,
            self.model.soc_info,
        )

        self.max_memory = NpuHbmInfo.get_hbm_capacity(
            self.local_rank, self.world_size, self.model.soc_info.need_nz
        )
        self.init_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, "
            f"init_memory(GB): {self.init_memory / (1024 ** 3)}",
        )

        self.warm_up_memory = 0
        self.warm_up_num_blocks = 0
        self.cache_manager = None

    def __repr__(self):
        return (
            f"PARunner("
            + f"model_path={self.model_path}, "
            + f"input_text={self.input_text}, "
            + f"max_position_embeddings={self.max_position_embeddings}, "
            + f"max_input_length={self.max_input_length}, "
            + f"max_output_length={self.max_output_length}, "
            + f"max_prefill_tokens={self.max_prefill_tokens}, "
            + f"is_flash_model={self.is_flash_model}, "
            + f"max_batch_size={self.max_batch_size}, "
            + f"use_refactor={self.use_refactor}, "
            + f"dtype={self.dtype}, "
            + f"block_size={self.block_size}, "
            + f"model_config={self.model_config}, "
            + f"max_memory={self.max_memory}, "
        )

    def warm_up(self):
        if self.max_prefill_tokens == -1:
            self.max_prefill_tokens = self.max_batch_size * (
                self.max_input_length + self.max_output_length
            )
        all_input_length = self.max_batch_size * self.max_input_length
        input_ids = torch.ones(all_input_length, dtype=torch.int64).to(self.device)
        position_ids = (
            torch.arange(self.max_input_length, dtype=torch.int32)
            .repeat(self.max_batch_size)
            .to(self.device)
        )
        cu_seqlen_prefill = torch.tensor([1])
        try:
            block_num = math.ceil(all_input_length / self.block_size)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        block_tables_tensor = (
            torch.arange(block_num, dtype=torch.int32).view(1, -1).to(self.device)
        )
        slots = torch.arange(all_input_length, dtype=torch.int32).to(self.device)
        input_lengths_tensor = torch.tensor(
            [self.max_input_length] * self.max_batch_size, dtype=torch.int64
        ).to(self.device)
        prefill_head_indices = torch.tensor(
            [all_input_length - 1], dtype=torch.int64
        ).to(self.device)
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        try:
            self.warm_up_num_blocks = (
                math.ceil(
                    (self.max_input_length + self.max_output_length) / self.block_size
                )
                * self.max_batch_size
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        cache_config = CacheConfig(self.warm_up_num_blocks, self.block_size)
        self.cache_manager = CacheManager(cache_config, self.model_config)
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=cu_seqlen_prefill is not None,
            block_tables=block_tables_tensor,
            kv_cache=self.cache_manager.kv_cache,
            slots=slots,
            input_lengths=input_lengths_tensor,
            max_seq_len=self.max_input_length,
            lm_head_indices=prefill_head_indices,
        )
        self.warm_up_memory = int(
            self.max_memory
            * NpuHbmInfo.get_hbm_usage(
                self.local_rank, self.world_size, self.model.soc_info.need_nz
            )
        )
        print_log(
            self.rank,
            logger.info,
            f"warmup_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}",
        )
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(
        self, input_texts, batch_size, max_output_length, ignore_eos, input_ids=None
    ):
        print_log(
            self.rank, logger.info, "---------------begin inference---------------"
        )
        if input_ids:
            if len(input_ids) == 1:
                req_list = [
                    request_from_token(
                        input_ids[0], max_output_length, self.block_size, req_idx=i
                    )
                    for i in range(batch_size)
                ]
            else:
                req_list = [
                    request_from_token(
                        input_ids, max_output_length, self.block_size, req_idx=i
                    )
                    for i, input_ids in enumerate(input_ids)
                ]
        else:
            if len(input_texts) == 1:
                req_list = [
                    request_from_text(
                        input_texts[0],
                        self.tokenizer,
                        max_output_length,
                        self.block_size,
                        req_idx=i,
                    )
                    for i in range(batch_size)
                ]
            else:
                req_list = [
                    request_from_text(
                        input_text,
                        self.tokenizer,
                        max_output_length,
                        self.block_size,
                        req_idx=i,
                    )
                    for i, input_text in enumerate(input_texts)
                ]
        print_log(
            self.rank, logger.debug, f"req_list[0].input_ids: {req_list[0].input_ids}"
        )

        if not self.cache_manager:
            if self.max_prefill_tokens == -1:
                self.max_prefill_tokens = self.max_batch_size * (
                    self.max_input_length + self.max_output_length
                )
            cache_block_size = (
                self.block_size * self.model.num_kv_heads * self.model.head_size
            )
            dtype_size = CacheManager.get_dtype_size(self.dtype)
            total_cache_size = self.model.num_layers * cache_block_size * 2 * dtype_size

            max_memory = (
                ENV.memory_fraction * self.max_memory
                if not ENV.max_memory_gb
                else int(ENV.max_memory_gb) * (1 << 30)
            )
            free_memory = (
                max_memory
                - ENV.reserved_memory_gb * (1 << 30)
                - (
                    self.warm_up_memory
                    if self.warm_up_memory != 0
                    else self.init_memory
                )
            )
            print_log(
                self.rank,
                logger.info,
                f"infer max_memory(GB): {max_memory / (1024 ** 3): .2f}, "
                f"warm_up_memory(GB): {self.warm_up_memory / (1024 ** 3): .2f}, "
                f"free_memory(GB): {free_memory / (1024 ** 3): .2f}",
            )

            num_blocks = int(free_memory // total_cache_size)
            print_log(
                self.rank,
                logger.info,
                f"num_blocks: {num_blocks}, free_memory: {free_memory}",
            )
            cache_config = CacheConfig(num_blocks, self.block_size)
            self.cache_manager = CacheManager(cache_config, self.model_config)

        if ENV.benchmark_enable:
            req_list_dummy = copy.deepcopy(req_list)
            generate_req(
                req_list_dummy,
                self.model,
                self.tokenizer,
                self.max_batch_size,
                self.max_prefill_tokens,
                2,
                self.cache_manager,
                self.rank,
                ignore_eos,
            )

        if not ENV.profiling_enable:
            print_log(self.rank, logger.debug, "no profiling")
            torch.npu.synchronize()
            e2e_start = time.time()
            generate_req(
                req_list,
                self.model,
                self.tokenizer,
                self.max_batch_size,
                self.max_prefill_tokens,
                max_output_length,
                self.cache_manager,
                self.rank,
                ignore_eos,
            )
            _, _ = decode_token(req_list, self.tokenizer)
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start
        else:
            print_log(self.rank, logger.debug, "enter profiling")
            import os

            profiling_path = ENV.profiling_filepath
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            torch.npu.synchronize()
            e2e_start = time.time()
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                l2_cache=False,
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    profiling_path
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ) as _:
                generate_req(
                    req_list,
                    self.model,
                    self.tokenizer,
                    self.max_batch_size,
                    self.max_prefill_tokens,
                    max_output_length,
                    self.cache_manager,
                    self.rank,
                    ignore_eos,
                )
            torch.npu.synchronize()
            e2e_end = time.time()
            e2e_time = e2e_end - e2e_start

        generate_text_list, token_num_list = decode_token(req_list, self.tokenizer)
        print_log(self.rank, logger.info, "---------------end inference---------------")
        return generate_text_list, token_num_list, e2e_time


def parse_ids(list_str):
    return [int(item) for item in list_str.split(",")]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        help="model and tokenizer path",
        default="/data/acltransformer_testdata/weights/llama2/llama-2-70b",
    )
    parser.add_argument(
        "--input_texts",
        type=str,
        nargs="+",
        default=[
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，请问 pandas 如何使用？给出一些代码示例来说明<|im_end|>\n<|im_start|>assistant\n"
        ],
    )
    parser.add_argument("--input_ids", type=parse_ids, nargs="+", default=None)
    parser.add_argument(
        "--input_file",
        type=str,
        help="CSV or Numpy file containing tokenized input. Alternative to text input.",
        default=None,
    )
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--max_prefill_tokens", type=int, default=-1)
    parser.add_argument("--max_batch_size", type=int, default=1)
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

    pa_runner = PARunner(**input_dict)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()

    generate_texts, token_nums, e2e_time = pa_runner.infer(
        args.input_texts,
        args.max_batch_size,
        args.max_output_length,
        args.ignore_eos,
        args.input_ids,
    )

    for i, generate_text in enumerate(generate_texts):
        length = len(args.input_ids) if args.input_ids else len(args.input_texts)
        inputs = args.input_ids if args.input_ids else args.input_texts
        if i < length:
            print_log(rank, logger.info, f"Question[{i}]: {inputs[i]}")
        print_log(rank, logger.info, f"Answer[{i}]: {generate_text}")
        print_log(rank, logger.info, f"Generate[{i}] token num: {token_nums[i]}")
        print_log(rank, logger.info, f"Cost seconds: {e2e_time}")

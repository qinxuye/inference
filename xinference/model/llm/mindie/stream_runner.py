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
import gc
import os
import time
import uuid
from typing import Dict, Iterable, Iterator, List, Optional, TypedDict, Union

import torch
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log

from ....device_utils import empty_cache
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    max_tokens_field,
)
from ..pytorch.utils import is_partial_stop
from ..utils import ChatModelMixin
from .runner import PARunner
from .server.cache import CacheConfig, CacheManager
from .server.generate import Batch, decode_token, generate_token
from .server.request import request_from_token


class MindIEModelConfig(TypedDict, total=False):
    pass


class MindIEGenerateConfig(TypedDict, total=False):
    pass


class StreamPARunner(PARunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for arg in [
            "model_uid",
            "model_family",
            "model_spec",
            "quantization",
            "model_path",
        ]:
            try:
                setattr(self, arg, kwargs[arg])
            except KeyError:
                continue

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> MindIEGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = MindIEGenerateConfig()
        return sanitized

    def generate(
        self, prompt: str, generate_config: Optional[MindIEGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        def generator_wrapper(
            prompt: str, generate_config: MindIEGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        print_log(
            self.rank,
            logger.debug,
            f"Enter generate, prompt: {prompt}, generate config: {generate_config}",
        )

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in self._generate_stream(
                prompt,
                generate_config,
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, generate_config)

    def _init_cache_manager(self):
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

    def _empty_cache_manager(self):
        self.cache_manager = None
        gc.collect()
        empty_cache()

    def _generate_stream(self, prompt: str, generate_config: MindIEGenerateConfig):
        print_log(
            self.rank, logger.info, "---------------begin inference---------------"
        )

        tokenizer = self.tokenizer
        model = self.model
        stream_interval = generate_config.get("stream_interval", 2)
        stream = generate_config.get("stream", False)

        len_prompt = len(prompt)

        max_input_length = int(generate_config.get("max_input_length", 0))
        max_output_length = int(
            generate_config.get("max_output_length", max_tokens_field.default)
        )
        echo = bool(generate_config.get("echo", False))
        stop_str = generate_config.get("stop", None)
        stop_token_ids = generate_config.get("stop_token_ids", None) or []
        stop_token_ids.append(tokenizer.eos_token_id)
        ignore_eos = bool(generate_config.get("ignore_eos", False))

        if ".modeling_qwen." in str(type(model)).lower():
            # TODO: hacky
            input_ids = tokenizer(prompt, allowed_special="all").input_ids
        else:
            input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        if max_input_length:
            input_ids = input_ids[-max_input_length:]
        input_echo_len = len(input_ids)

        # init cache manager
        self._init_cache_manager()

        torch.npu.synchronize()
        start = time.time()
        token = None
        last_output_length = 0

        req = request_from_token(
            input_ids, max_output_length, self.block_size, req_idx=0
        )
        batch = Batch([req])

        for i in range(max_output_length):
            if i == 0:
                # prefill
                self.cache_manager.allocate(batch)
                prefill_start = time.time()
                torch.npu.synchronize()
                req_finished = generate_token(
                    self.model,
                    self.tokenizer,
                    self.cache_manager,
                    batch,
                    max_output_length,
                    self.rank,
                    ignore_eos,
                )
                torch.npu.synchronize()
                print_log(
                    self.rank,
                    logger.debug,
                    f"prefill time {time.time() - prefill_start} seconds",
                )
            else:
                # generate
                req_finished = generate_token(
                    self.model,
                    self.tokenizer,
                    self.cache_manager,
                    batch,
                    max_output_length,
                    rank,
                    ignore_eos,
                )
            token = req.out_token_list[-1]
            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_output_length - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = is_partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = is_partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                if stream:
                    output = output.strip("�")
                    tmp_output_length = len(output)
                    output = output[last_output_length:]
                    last_output_length = tmp_output_length

                # prevent yielding partial stop sequence
                if not partially_stopped:
                    completion_choice = CompletionChoice(
                        text=output, index=0, logprobs=None, finish_reason=None
                    )
                    completion_chunk = CompletionChunk(
                        id=str(uuid.uuid1()),
                        object="text_completion",
                        created=int(time.time()),
                        model=self.model_uid,
                        choices=[completion_choice],
                    )
                    completion_usage = CompletionUsage(
                        prompt_tokens=input_echo_len,
                        completion_tokens=i,
                        total_tokens=(input_echo_len + i),
                    )

                    yield completion_chunk, completion_usage

            if stopped:
                break

        elapsed_time = time.time() - start
        print_log(
            self.rank,
            logger.info,
            f"Average generation speed: {i / elapsed_time:.2f} tokens/s.",
        )

        # finish stream event, which contains finish reason
        if stopped:
            finish_reason = "stop"
        elif i == max_output_length - 1:
            finish_reason = "length"
        else:
            finish_reason = None

        if stream:
            completion_choice = CompletionChoice(
                text="", index=0, logprobs=None, finish_reason=finish_reason
            )
        else:
            completion_choice = CompletionChoice(
                text=output, index=0, logprobs=None, finish_reason=finish_reason
            )

        completion_chunk = CompletionChunk(
            id=str(uuid.uuid1()),
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=input_echo_len,
            completion_tokens=i,
            total_tokens=(input_echo_len + i),
        )

        yield completion_chunk, completion_usage

        # clean
        del output
        self._empty_cache_manager()


class StreamPAChatRunner(StreamPARunner, ChatModelMixin):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[MindIEGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style, tools=tools)

        generate_config = self._sanitize_generate_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in ["qwen-chat", "qwen1.5-chat"]:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = list(stop) + ["Observation:"]
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)


def parse_ids(list_str):
    return [int(item) for item in list_str.split(",")]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_uid",
        help="model UID",
        default="qwen1.5-chat",
    )
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

    pa_runner = StreamPARunner(**input_dict)
    print_log(rank, logger.info, f"pa_runner: {pa_runner}")
    pa_runner.warm_up()
    completion = pa_runner.generate(args.input_texts[0])
    print_log(rank, logger.info, f'output: {repr(completion["choices"][0]["text"])}')

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

import gc
import weakref
from typing import Iterable, List

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log

from ....device_utils import empty_cache
from ....types import max_tokens_field
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..transformers.utils import get_context_length
from ..utils import ChatModelMixin
from .config import MindIEModelConfig
from .scheduler import InferenceRequest
from .server.cache import CacheConfig, CacheManager
from .server.runner import PARunner


class StreamPARunner(PARunner):
    model_uid: str
    model_family: "LLMFamilyV1"
    model_spec: "LLMSpecV1"
    quantization: str
    model_path: str
    _model_config: MindIEModelConfig
    _context_len: int
    _req_to_batch: weakref.WeakKeyDictionary

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
        self._model_config = kwargs.pop("model_config")
        self._context_len = self._model_config["context_length"] or get_context_length(
            self.model.config
        )
        self._req_to_batch = weakref.WeakKeyDictionary()

    def get_max_num_seqs(self) -> int:
        return self.max_batch_size

    def init_cache_manager(self):
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

    def empty_cache_manager(self):
        self.cache_manager = None
        gc.collect()
        empty_cache()

    def _get_full_prompt(self, prompt, system_prompt, chat_history, tools):
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = ChatModelMixin.get_prompt(
            prompt, chat_history, prompt_style, tools=tools
        )
        return full_prompt

    def batch_inference(self, req_list: List[InferenceRequest]):
        from .utils import batch_inference_one_step

        for r in req_list:
            if r.sanitized_generate_config is None:
                r.sanitized_generate_config = r.generate_config
            if r.is_prefill:
                # check some generate params
                max_new_tokens = r.sanitized_generate_config.get(
                    "max_output_length", max_tokens_field.default
                )
                max_src_len = self._context_len - max_new_tokens - 8
                if max_src_len < 0:
                    r.stopped = True
                    r.error_msg = "Max tokens exceeds model's max length"
                    continue
                if r.stream_interval <= 0:
                    r.stopped = True
                    r.error_msg = "`stream_interval` must be greater than 0"
                    continue
                stop_str = r.sanitized_generate_config.get("stop", None)
                if stop_str and (
                    not (isinstance(stop_str, str) or isinstance(stop_str, Iterable))
                ):
                    r.stopped = True
                    r.error_msg = "Invalid `stop` field type"
                    continue
                r.full_prompt = self._get_full_prompt(
                    r.prompt, r.system_prompt, r.chat_history, None
                )

        assert isinstance(self._context_len, int)
        batch_inference_one_step(
            req_list,
            self.model_uid,
            self.model,
            self.tokenizer,
            self._context_len,
            self.rank,
            self.world_size,
            self.block_size,
            self.cache_manager,
            self._req_to_batch,
        )
        for req in req_list:
            if req.error_msg is None and req.completion:
                if req.stream:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            results.append(
                                ChatModelMixin._get_first_chat_completion_chunk(
                                    req.completion[i + 1]
                                )
                            )
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(ChatModelMixin._to_chat_completion_chunk(c))

                    if req.stopped and req.include_usage:
                        results.append(
                            ChatModelMixin._get_final_chat_completion_chunk(
                                req.completion[-1]
                            )
                        )
                    req.completion = results
                else:
                    req.completion[0] = ChatModelMixin._to_chat_completion(
                        req.completion[0]
                    )

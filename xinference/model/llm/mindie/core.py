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
import json
import logging
import os.path
import subprocess
import time
import uuid
from typing import AsyncGenerator, Dict, Iterable, List, Optional, Union

import psutil
import xoscar as xo
from xoscar.utils import get_next_port

from ....device_utils import gpu_count, is_device_available
from ....isolation import Isolation
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    LoRA,
)
from ...utils import create_symlink
from ..core import LLM
from ..llm_family import CustomLLMFamilyV1, LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin
from .config import MindIEGenerateConfig, MindIEModelConfig

try:
    import atb_llm  # noqa: F401

    MINDIE_INSTALLED = True
except ImportError:
    MINDIE_INSTALLED = False


logger = logging.getLogger(__name__)


MINDIE_SUPPORTED_MODELS: List[str] = []
MINDIE_SUPPORTED_CHAT_MODELS: List[str] = [
    "baichuan-chat",
    "baichuan-2-chat",
    "chatglm3",
    "deepseek-chat",
    "deepseek-coder-instruct",
    "llama-3-instruct",
    "mistral-instruct-v0.3",
    "telechat",
    "Yi-chat",
    "Yi-1.5-chat",
    "qwen-chat",
    "qwen1.5-chat",
    "codeqwen1.5-chat",
    "qwen2-instruct",
]


class MindIEModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[MindIEModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        if peft_model is not None:
            raise ValueError("Peft model is not supported for MindIE")

        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._model_config = model_config
        self._client = None
        self._process: Optional[subprocess.Popen] = None

    def _ensure_model_compatible(self) -> str:
        model_path = self.model_path
        mindie_path = "-".join([model_path, "mindie"])
        mindie_config_path = os.path.join(mindie_path, "config.json")
        if os.path.exists(mindie_path) and os.path.exists(mindie_config_path):
            # the path is converted already, check if it can use
            with open(os.path.join(mindie_path, "config.json")) as f:
                params = json.load(f)
                if params.get("torch_dtype") == "float16":
                    return mindie_path

        model_config_file = os.path.join(model_path, "config.json")
        with open(model_config_file) as f:
            params = json.load(f)
            torch_dtype = params.get("torch_dtype")
            assert torch_dtype in [
                "float16",
                "bfloat16",
                None,
            ], "torch_dtype can only be None, float16 or bfloat16"
            if torch_dtype == "float16":
                # MindIE can only support float16
                return model_path

            logger.info(
                "Model path is not compatible with MindIE when torchdtype=%s",
                torch_dtype,
            )

            os.makedirs(mindie_path, exist_ok=True)

            create_symlink(model_path, mindie_path)
            # delete config.json
            os.remove(os.path.join(mindie_path, "config.json"))
            # write new config.json
            with open(os.path.join(mindie_path, "config.json"), "w") as cf:
                params["torch_dtype"] = "float16"
                json.dump(params, cf, indent=2)
            return mindie_path

    def load(self):
        from .main import MindIEModelActor

        self._model_config = self._sanitize_model_config(self._model_config)
        port = get_next_port()
        # launch PAStreamRunner
        commands = [
            "torchrun",
            "--nproc_per_node",
            str(self._model_config["tensor_parallel_size"]),
            "--master_port",
            str(get_next_port()),
            "-m",
            "xinference.model.llm.mindie.main",
            "--port",
            str(port),
            "--max_batch_size",
            str(self._model_config["max_batch_size"]),
        ]
        logger.info("Launch MindIE with command, %s", commands)
        self._process = process = subprocess.Popen(commands)
        model_path = self._ensure_model_compatible()

        async def try_connect():
            client: xo.ActorRefType[MindIEModelActor] = await xo.actor_ref(  # type: ignore
                f"127.0.0.1:{port}", MindIEModelActor.uid()
            )
            await client.update_kwargs(
                model_uid=self.model_uid,
                model_family=self.model_family,
                model_spec=self.model_spec,
                quantization=self.quantization,
                model_path=model_path,
                model_config=self._model_config,
            )
            await client.wait_ready()
            self._client = client

        isolation = Isolation(asyncio.new_event_loop())
        isolation.start()
        try:
            while True:
                status = process.poll()
                if status is not None and status != 0:
                    raise RuntimeError("fail to load model")
                try:
                    isolation.call(try_connect())
                except (ConnectionError, xo.ActorNotExist) as e:
                    logger.debug("Connect to MindIE process failed, %s", e)
                    time.sleep(3)
                except Exception as e:
                    try:
                        self.stop()
                    except:
                        # ignore stop error
                        pass
                    raise RuntimeError("Fail to load model") from e
                else:
                    break
        finally:
            isolation.stop()

    def stop(self):
        if self._process:
            # kill all the subprocesses created by torchrun
            pid = self._process.pid
            for p in psutil.Process(pid).children(recursive=True):
                p.kill()
            self._process.kill()

    def _sanitize_model_config(
        self, model_config: Optional[MindIEModelConfig]
    ) -> MindIEModelConfig:
        if model_config is None:
            model_config = MindIEModelConfig()

        cuda_count = gpu_count()
        model_config.setdefault("tensor_parallel_size", cuda_count)
        model_config.setdefault("context_length", 4096)
        model_config.setdefault("max_batch_size", 15)

        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> MindIEGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = MindIEGenerateConfig()
        sanitized.setdefault(
            "max_output_length", generate_config.get("max_tokens", 1024)
        )
        sanitized.setdefault(
            "repetition_penalty", generate_config.get("repetition_penalty", 1.1)
        )
        sanitized.setdefault("temperature", generate_config.get("temperature", 1.0))
        sanitized.setdefault("top_p", generate_config.get("top_p", 1.0))
        sanitized.setdefault("top_k", generate_config.get("top_k", -1))
        sanitized.setdefault("stop", generate_config.get("stop", None))
        sanitized.setdefault(
            "stop_token_ids", generate_config.get("stop_token_ids", None)
        )
        sanitized.setdefault(
            "stream_interval", generate_config.get("stream_interval", 2)
        )
        sanitized.setdefault("stream", generate_config.get("stream", False))
        sanitized.setdefault(
            "stream_options", generate_config.get("stream_options", None)
        )

        return sanitized

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._is_linux():
            return False
        if not is_device_available("npu"):
            return False
        if llm_spec.model_format not in ["pytorch"]:
            return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in MINDIE_SUPPORTED_MODELS:
                return False
        else:
            if llm_family.model_name not in MINDIE_SUPPORTED_MODELS:
                return False
        if "generate" not in llm_family.model_ability:
            return False
        return MINDIE_INSTALLED

    async def async_generate(
        self,
        prompt: str,
        generate_config: Optional[Dict] = None,
        tools: object = False,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        stream = sanitized_generate_config.pop("stream")

        async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
            async for chunk in await self._client.generate(  # type: ignore
                prompt, sanitized_generate_config
            ):
                yield chunk

        if stream:
            return stream_results()
        else:
            return await self._client.generate(prompt, sanitized_generate_config)  # type: ignore


class MindIEChatModel(MindIEModel, ChatModelMixin):
    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._is_linux():
            return False
        if not is_device_available("npu"):
            return False
        if llm_spec.model_format not in ["pytorch"]:
            return False
        if isinstance(llm_family, CustomLLMFamilyV1):
            if llm_family.model_family not in MINDIE_SUPPORTED_CHAT_MODELS:
                return False
        else:
            if llm_family.model_name not in MINDIE_SUPPORTED_CHAT_MODELS:
                return False
        if "chat" not in llm_family.model_ability:
            return False
        return MINDIE_INSTALLED

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}
        generate_config.setdefault(
            "max_output_length", generate_config.get("max_tokens", 1024)
        )
        if self.model_family.prompt_style:
            if (
                not generate_config.get("stop")
            ) and self.model_family.prompt_style.stop:
                generate_config["stop"] = self.model_family.prompt_style.stop.copy()
            if self.model_family.prompt_style.stop_token_ids:
                generate_config.setdefault(
                    "stop_token_ids",
                    self.model_family.prompt_style.stop_token_ids.copy(),
                )
        return generate_config

    async def async_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        tools = generate_config.pop("tools", []) if generate_config else None

        generate_config = self._sanitize_chat_config(generate_config)
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

        stream = generate_config.get("stream", None)
        if "request_id" not in generate_config:
            # generate uuid if not provided
            generate_config["request_id"] = str(uuid.uuid4().hex)

        async def stream_results() -> AsyncGenerator[ChatCompletionChunk, None]:
            async for chunk in await self._client.chat(  # type: ignore
                prompt, system_prompt, chat_history, generate_config
            ):
                yield chunk

        if stream:
            return stream_results()
        else:
            c = await self._client.chat(prompt, system_prompt, chat_history, generate_config)  # type: ignore
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return c

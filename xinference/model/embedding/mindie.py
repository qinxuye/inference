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
import gc
import logging
import warnings
from typing import Union, List

from .core import EmbeddingModel, EMBEDDING_EMPTY_CACHE_COUNT
from ...types import Embedding, EmbeddingData, EmbeddingUsage
from ...device_utils import empty_cache, is_device_available

logger = logging.getLogger(__file__)


class MindIEEmbeddingModel(EmbeddingModel):
    def load(self):
        from ais_bench.infer.interface import InferSession
        from transformers import AutoTokenizer

        model_path = self._model_path
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        om_files = [f for f in os.listdir(model_path) if f.endswith('.om')]
        if not om_files:
            raise ValueError(f"No .om files found in {model_path}")
        om_file_name = om_files[0]
        om_model_path = os.path.join(model_path, om_file_name)
        self._session = InferSession(device_id=self._device or 0, model_path=om_model_path)

    def create_embedding(self, sentences: Union[str, List[str]], **kwargs):
        import numpy as np
        import torch

        self._counter += 1
        if self._counter % EMBEDDING_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty embedding cache.")
            gc.collect()
            empty_cache()

        raw_sentences = sentences
        if isinstance(raw_sentences, str):
            sentences = [sentences]

        all_token_nums = 0
        encoded_input = self._tokenizer(sentences, padding=True, truncation=True, return_tensors='np', max_length=self._model_spec.max_tokens)
        input_ids = encoded_input['input_ids']
        all_token_nums += sum(len(i) for i in input_ids)
        attention_mask = encoded_input['attention_mask']
        token_type_ids = encoded_input['token_type_ids']
        inputs = [input_ids, attention_mask, token_type_ids]
        outputs = self._session.infer(feeds=inputs, mode="dymshape", custom_sizes=10000000)[0][:, 0]
        outputs = torch.from_numpy(outputs)
        all_embeddings = torch.nn.functional.normalize(outputs, p=2, dim=1)

        embedding_list = []
        for index, data in enumerate(all_embeddings):
            embedding_list.append(
                EmbeddingData(index=index, object="embedding", embedding=data.tolist())
            )
        usage = EmbeddingUsage(
            prompt_tokens=all_token_nums, total_tokens=all_token_nums
        )
        return Embedding(
            object="list",
            model=self._model_uid,
            data=embedding_list,
            usage=usage,
        )


def is_available(model_spec) -> bool:
    if not is_device_available("npu"):
        return False
    if "bge-large-zh-v1.5" not in getattr(model_spec, "model_id", getattr(model_spec, "model_name", "")):
        raise False
    try:
        from ais_bench.infer.interface import InferSession
    except Exception as e:
        warnings.warn(f"Cannot run accelerated embedding inference, reason: {e}")
        return False
    else:
        return True

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

from typing import List, Optional, TypedDict, Union


class MindIEModelConfig(TypedDict, total=False):
    context_length: int
    tensor_parallel_size: int
    max_batch_size: int


class MindIEGenerateConfig(TypedDict, total=False):
    max_input_length: int
    max_output_length: int
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int
    stop: str
    stop_token_ids: List[int]
    stream_interval: int
    stream: bool
    stream_options: Optional[Union[dict, None]]

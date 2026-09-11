# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
from unittest.mock import MagicMock

import pytest

from ..restful import async_restful_client as client_module


@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
def test_destructor_does_not_require_an_event_loop(client_class, monkeypatch):
    def fail_get_event_loop():
        raise AssertionError("destructor must not request an implicit event loop")

    monkeypatch.setattr(client_module.asyncio, "get_event_loop", fail_get_event_loop)
    monkeypatch.setattr(
        client_module.asyncio,
        "get_running_loop",
        MagicMock(side_effect=RuntimeError("no running event loop")),
    )
    client = client_class.__new__(client_class)
    client.session = object()

    client.__del__()

    client.session = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
async def test_destructor_schedules_close_on_running_loop(client_class):
    client = client_class.__new__(client_class)
    client.session = object()
    closed = asyncio.Event()

    async def close():
        client.session = None
        closed.set()

    client.close = close

    client.__del__()

    await asyncio.wait_for(closed.wait(), timeout=1)
    assert client.session is None

# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

spec = importlib.util.spec_from_file_location(
    "pd_affinity_gpu", Path(__file__).with_name("test_pd_affinity_gpu.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gpu_guard_distinguishes_owned_idle_and_foreign_processes(monkeypatch):
    monkeypatch.setenv("XINFERENCE_TEST_PD_IDLE_GPU_PIDS", "101")
    monkeypatch.setattr(module.os, "getpid", lambda: 100)
    parent = Mock()
    parent.children.return_value = [Mock(pid=102)]
    monkeypatch.setattr(psutil, "Process", lambda: parent)
    monkeypatch.setattr(
        module.subprocess, "run", Mock(return_value=Mock(stdout="100\n101\n102\n103\n"))
    )
    sample = module._gpu_snapshot()
    assert sample["foreign_pids"] == [103]
    assert sample["allowed_idle_pids"] == [101]


@pytest.mark.asyncio
async def test_gpu_guard_failure_invalidates_measurement(monkeypatch):
    def unavailable():
        raise RuntimeError("GPU monitoring unavailable")

    async def measure(*args):
        return [], 1.0, []

    monkeypatch.setattr(module, "_gpu_snapshot", unavailable)
    with pytest.raises(RuntimeError, match="monitoring unavailable"):
        await module._guarded_measure(Mock(measure=measure), "endpoint", "model", [], 8)

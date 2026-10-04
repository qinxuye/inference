# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..pd_model import PDModelActor, PrefixAffinitySchedulingPolicy


def test_prefix_affinity_and_busy_fallback():
    policy = PrefixAffinitySchedulingPolicy(["p0", "p1"])
    policy.remember(b"a", "p0")
    assert policy.schedule(b"a") == "p0"
    policy.started("p0", "a")
    assert policy.schedule(b"a") == "p0"
    policy.started("p0", "b")
    assert policy.schedule(b"a") == "p1"
    policy.finished("p0", "a")
    policy.finished("p0", "b")
    assert policy.schedule(b"a") == "p0"
    policy.remember(b"a", "p1")
    assert {policy.schedule(b"a") for _ in range(2)} == {"p0", "p1"}


def test_hints_are_bounded_and_removed_with_replica():
    policy = PrefixAffinitySchedulingPolicy(["p0", "p1"], capacity=2)
    for key in (b"a", b"b", b"c"):
        policy.remember(key, "p0")
    assert list(policy._prefixes) == [b"b", b"c"]
    policy.started("p0", "old")
    policy.update_replicas(["p1", "replacement"])
    policy.finished("p0", "old")
    policy.remember(b"old-completion", "p0")
    assert not policy._prefixes
    assert policy._pending == {"p1": set(), "replacement": set()}


def test_prefix_key_ignores_suffix_and_skips_short_or_nontext_inputs():
    key = PrefixAffinitySchedulingPolicy.prefix_key
    prefix = "Long shared context. " * 100
    assert key("chat", [{"role": "user", "content": prefix + "one"}]) == key(
        "chat", [{"role": "user", "content": prefix + "two"}]
    )
    assert key("generate", prefix) != key("generate", "Different prefix. " + prefix)
    assert key("generate", "short") is None
    assert key("chat", [{"role": "user", "content": [{"type": "image_url"}]}]) is None
    assert key("chat", []) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "prefill", "decode", "cancel"])
async def test_router_learns_only_success_and_releases_prefill_load(failure):
    actor = PDModelActor("test", scheduling_policy=PrefixAffinitySchedulingPolicy)
    p, d = MagicMock(), MagicMock()
    p.chat = AsyncMock(
        return_value={"_pd_kv_transfer_params": {"do_remote_prefill": True}}
    )
    d.chat = AsyncMock(return_value=b'{"choices":[]}')
    await actor.add_prefill_actor("p", p)
    await actor.add_decode_actor("d", d)
    prompt = [{"role": "user", "content": "context " * 200}]
    if failure:
        error = asyncio.CancelledError if failure == "cancel" else RuntimeError
        (d if failure == "decode" else p).chat.side_effect = error()
        with pytest.raises(error):
            await actor._infer("chat", prompt, {})
    else:
        await actor._infer("chat", prompt, {})
    assert actor._prefill_policy._pending[p] == set()
    assert bool(actor._prefill_policy._prefixes) is (failure is None)


@pytest.mark.asyncio
async def test_prefill_load_is_visible_until_prefill_finishes():
    actor = PDModelActor("test", scheduling_policy=PrefixAffinitySchedulingPolicy)
    entered, release = asyncio.Event(), asyncio.Event()
    p, d = MagicMock(), MagicMock()

    async def prefill(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"_pd_kv_transfer_params": {"do_remote_prefill": True}}

    p.chat = prefill
    d.chat = AsyncMock(return_value=b'{"choices":[]}')
    await actor.add_prefill_actor("p", p)
    await actor.add_decode_actor("d", d)
    task = asyncio.create_task(actor._infer("chat", [], {}))
    await entered.wait()
    assert len(actor._prefill_policy._pending[p]) == 1
    release.set()
    await task
    assert actor._prefill_policy._pending[p] == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_stream_learns_only_after_successful_completion(complete):
    actor = PDModelActor("test", scheduling_policy=PrefixAffinitySchedulingPolicy)
    p, d = MagicMock(), MagicMock()
    p.chat = AsyncMock(
        return_value={"_pd_kv_transfer_params": {"do_remote_prefill": True}}
    )

    async def chunks():
        yield b"first"
        yield b"second"

    d.chat = AsyncMock(return_value=chunks())
    d.decrease_serve_count = AsyncMock()
    await actor.add_prefill_actor("p", p)
    await actor.add_decode_actor("d", d)
    stream = await actor._infer(
        "chat", [{"role": "user", "content": "context " * 200}], {}
    )
    assert len(actor._prefill_policy._pending[p]) == 1
    assert await anext(stream) == b"first"
    assert not actor._prefill_policy._pending[p]
    assert not actor._prefill_policy._prefixes
    if complete:
        assert [chunk async for chunk in stream] == [b"second"]
    else:
        await stream.aclose()
    assert bool(actor._prefill_policy._prefixes) is complete
    assert actor._prefill_policy._pending[p] == set()


def test_old_completion_cannot_release_readded_replica_request():
    policy = PrefixAffinitySchedulingPolicy(["p0", "p1"])
    policy.started("p0", "old")
    policy.update_replicas(["p1"])
    policy.update_replicas(["p1", "p0"])
    policy.started("p0", "new")
    policy.finished("p0", "old")
    assert policy._pending["p0"] == {"new"}
    policy.finished("p0", "new")
    assert not policy._pending["p0"]


@pytest.mark.asyncio
async def test_abort_before_stream_iteration_releases_inflight_handoff():
    actor = PDModelActor("test", scheduling_policy=PrefixAffinitySchedulingPolicy)
    p, d = MagicMock(), MagicMock()
    p.chat = AsyncMock(
        return_value={"_pd_kv_transfer_params": {"do_remote_prefill": True}}
    )

    async def chunks():
        yield b"first"

    d.chat = AsyncMock(return_value=chunks())
    for ref in (p, d):
        ref.abort_request = AsyncMock(return_value="DONE")
    await actor.add_prefill_actor("p", p)
    await actor.add_decode_actor("d", d)
    stream = await actor._infer("chat", [], {}, request_id="abort-before-read")
    assert actor._prefill_policy._pending[p] == {"abort-before-read"}
    await actor.abort_request("abort-before-read")
    assert not actor._prefill_policy._pending[p]
    assert not actor._prefill_inflight
    await stream.aclose()

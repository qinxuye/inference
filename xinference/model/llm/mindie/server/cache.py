# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os

import torch
from atb_llm.utils.log import logger


class CacheConfig:
    def __init__(self, num_blocks=1024, block_size=128):
        self.num_blocks = int(os.getenv("NUM_BLOCKS", f"{num_blocks}"))
        self.block_size = int(os.getenv("BLOCK_SIZE", f"{block_size}"))


class ModelConfig:
    def __init__(
        self, num_heads, num_kv_heads, head_size, num_layers, device, dtype, soc_info
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.soc_info = soc_info

    def __repr__(self):
        return (
            f"ModelConfig("
            + f"num_heads={self.num_heads}, "
            + f"num_kv_heads={self.num_kv_heads}, "
            + f"head_size={self.head_size}, "
            + f"num_layers={self.num_layers}, "
            + f"device={self.device}, "
            + f"dtype={self.dtype}, "
            + f"soc_info={self.soc_info}, "
        )


class CacheManager:
    def __init__(self, cache_config, model_config):
        self.block_size = cache_config.block_size
        self.num_blocks = cache_config.num_blocks

        self.num_heads = model_config.num_kv_heads
        self.head_size = model_config.head_size
        self.num_layers = model_config.num_layers
        self.device = model_config.device
        self.dtype = model_config.dtype
        self.soc_info = model_config.soc_info

        mem_need = (
            self.num_blocks
            * self.block_size
            * self.num_heads
            * self.head_size
            * self.num_layers
            * 2
            * self.get_dtype_size(self.dtype)
            / 1024
            / 1024
            / 1024
        )
        logger.info(f"kv cache will allocate {mem_need}GB memory")

        if self.soc_info.need_nz:
            self.kv_cache = [
                (
                    torch.empty(
                        (
                            self.num_blocks,
                            self.num_heads * self.head_size // 16,
                            self.block_size,
                            16,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        (
                            self.num_blocks,
                            self.num_heads * self.head_size // 16,
                            self.block_size,
                            16,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for _ in range(self.num_layers)
            ]
        else:
            self.kv_cache = [
                (
                    torch.empty(
                        (
                            self.num_blocks,
                            self.block_size,
                            self.num_heads,
                            self.head_size,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        (
                            self.num_blocks,
                            self.block_size,
                            self.num_heads,
                            self.head_size,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for _ in range(self.num_layers)
            ]

        random_block_allocate = os.getenv("RANDOM_BLOCK_ALLOCATE", "0") == "1"
        if random_block_allocate:
            self.block_map = torch.randperm(self.num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.zeros(self.num_blocks, dtype=torch.long)
            for i in range(self.num_blocks):
                self.contrary_block_map[self.block_map[i]] = i
        else:
            self.block_map = torch.arange(self.num_blocks, dtype=torch.long)
            self.contrary_block_map = torch.arange(self.num_blocks, dtype=torch.long)

        self.free_block_mask = torch.ones(self.num_blocks, dtype=torch.long)
        self.total_slots = torch.arange(
            self.num_blocks * self.block_size, dtype=torch.long
        )
        self.total_slots = self.total_slots.view(self.num_blocks, self.block_size)

    @staticmethod
    def get_dtype_size(dtype):
        dtype_size_map = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2}
        return dtype_size_map.get(dtype, 2)

    def allocate(self, batch):
        total_need_blocks = 0
        max_need_blocks = 0
        for req in batch.req_list:
            if req.block_tables:
                logger.error(f"req_id: {req.req_id} block has been allocated")
                raise AssertionError

            total_need_blocks += req.need_blocks
            max_need_blocks = max(max_need_blocks, req.need_blocks)

        free_block_indices = self.free_block_mask.nonzero().flatten()
        if free_block_indices.numel() < total_need_blocks:
            logger.error(
                f"Out of available cache blocks: asked {total_need_blocks}, "
                f"only {free_block_indices.numel()} free blocks"
            )
            raise AssertionError

        allocate_block_indices = free_block_indices[:total_need_blocks]
        allocate_blocks = self.block_map[allocate_block_indices]

        block_offset = 0
        block_tables_list = []
        slot_tables_list = []
        for req in batch.req_list:
            req.block_tables = allocate_blocks[
                block_offset : block_offset + req.need_blocks
            ]
            req.slot_tables = self.total_slots[req.block_tables].flatten()
            block_tables = req.block_tables
            if req.need_blocks < max_need_blocks:
                block_tables = torch.concat(
                    [
                        block_tables,
                        torch.zeros(
                            max_need_blocks - req.need_blocks, dtype=torch.long
                        ),
                    ],
                    dim=0,
                )
            block_tables_list.append(block_tables.view(1, -1))
            slot_tables_list.append(req.slot_tables)
            block_offset += req.need_blocks

        batch.batch_block_tables = torch.concat(block_tables_list, dim=0)
        batch.batch_slots_tables = torch.concat(slot_tables_list, dim=0)

        self.free_block_mask[allocate_block_indices] = 0

    def free(self, req):
        if req.block_tables is not None:
            block_indices = self.contrary_block_map[req.block_tables]
            self.free_block_mask[block_indices] = 1

    def get_free_block_num(self):
        free_block_indices = self.free_block_mask.nonzero()
        return len(free_block_indices)

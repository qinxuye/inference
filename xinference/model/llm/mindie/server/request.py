# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import math
from typing import List

import torch


class Request:
    req_id: int

    input_ids: torch.Tensor
    input_length: int

    need_blocks: int
    need_slots: int
    block_tables: torch.Tensor
    slot_tables: torch.Tensor

    out_token_list: List[int]

    def __init__(
        self, max_out_length: int, block_size: int, req_id: int, input_ids: torch.Tensor
    ):
        self.req_id = req_id
        self.input_ids = input_ids.flatten()

        self.input_length = self.input_ids.numel()

        try:
            self.need_blocks = math.ceil(
                (self.input_length + max_out_length) / block_size
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        self.need_slots = self.need_blocks * block_size
        self.block_tables = None
        self.slot_tables = None

        self.out_token_list = []


def request_from_token(input_ids, max_out_length, block_size, req_idx=0) -> Request:
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    request = Request(max_out_length, block_size, req_idx, input_ids)
    return request


def request_from_text(
    text, tokenizer, max_out_length, block_size, req_idx=0
) -> Request:
    input_ids = tokenizer([text], return_tensors="pt")["input_ids"].flatten()
    request = request_from_token(input_ids, max_out_length, block_size, req_idx)
    return request


def request_from_token_file(input_path, max_out_length, block_size) -> List[Request]:
    req_list = []
    req_idx = 0
    with open(input_path, "r") as f:
        for line in f.readlines():
            token_str_list = line.split(",")
            input_ids = []
            for token_str in token_str_list:
                input_ids.append(int(token_str))
            req_list.append(
                request_from_token(input_ids, max_out_length, block_size, req_idx)
            )
            req_idx += 1
    return req_list


def request_from_text_file(
    input_path, tokenizer, max_out_length, block_size
) -> List[Request]:
    req_list = []
    req_idx = 0
    with open(input_path, "r") as f:
        for line in f.readlines():
            if line[-1] != "\n":
                continue
            text = line[:-1]
            req_list.append(
                request_from_text(
                    text, tokenizer, max_out_length, block_size, req_idx=0
                )
            )
            req_idx += 1
    return req_list

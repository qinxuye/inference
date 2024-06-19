# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pandas as pd
import torch
from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log

from .batch import Batch


def next_token_chooser(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1)


def generate_token(
    model,
    cache_manager,
    batch: Batch,
    rank,
    get_next_token_methods=None,
):
    input_ids = batch.batch_input_ids.npu()
    position_ids = batch.batch_position_ids.npu()
    is_prefill = batch.cu_seqlen_prefill is not None
    block_tables = batch.batch_block_tables.npu()
    kv_cache = cache_manager.kv_cache
    slots = batch.batch_slots_tables[batch.batch_slot_indices].npu()
    input_lengths = batch.context_length.npu()
    lm_head_indices = (
        None if batch.lm_head_indices is None else batch.lm_head_indices.npu()
    )

    logits = model.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        is_prefill=is_prefill,
        block_tables=block_tables,
        kv_cache=kv_cache,
        slots=slots,
        input_lengths=input_lengths,
        max_seq_len=batch.max_s,
        lm_head_indices=lm_head_indices,
    )

    if batch.cu_seqlen_prefill is not None and logits.size(0) != batch.batch_num:
        if logits.size(0) != batch.lm_head_indices[-1] + 1:
            logger.error(
                f"prefill logits is invalid, batch num: {batch.batch_num},"
                + f" total token: {int(batch.lm_head_indices[-1] + 1)}, but logits shape is: {logits.shape}"
            )
            raise AssertionError
        logits = logits[batch.lm_head_indices]

    ENV.update()
    if ENV.logits_save_enable:
        import os

        if rank == 0:
            logits_save_filename = (
                "logits_" + str(len(batch.req_list[0].out_token_list)) + ".pth"
            )
            torch.save(
                logits.cpu(), os.path.join(ENV.logits_save_folder, logits_save_filename)
            )
    if get_next_token_methods is None:
        next_token = next_token_chooser(logits)
        next_token_list = next_token.tolist()
    else:
        next_token_list = []
        for i, logit in enumerate(logits):
            next_token_list.append(get_next_token_methods[i](logit))
        next_token = torch.tensor(next_token_list)

    for i, req in enumerate(batch.req_list):
        req.out_token_list.append(next_token_list[i])

    batch.batch_input_ids = next_token.to(torch.int64)
    batch.batch_position_ids = batch.context_length.clone().to(torch.long)
    if batch.cu_seqlen_prefill is not None:
        batch.batch_slot_indices = batch.batch_slot_indices[batch.lm_head_indices]
        batch.cu_seqlen_prefill = None
        batch.lm_head_indices = None

    batch.batch_slot_indices += 1
    batch.context_length += 1
    batch.max_s += 1

    # eos_token_id = tokenizer.eos_token_id
    # return batch.filter(eos_token_id, max_out_length, cache_manager)


def generate_req(
    req_list,
    model,
    tokenizer,
    max_batch_size,
    max_prefill_tokens,
    max_out_length,
    cache_manager,
    rank,
    ignore_eos,
):
    req_num = len(req_list)
    print_log(rank, logger.info, f"------total req num: {req_num}, infer start--------")

    req_idx = 0
    total_req_finished = 0
    generate_batch_size = 0
    max_generate_batch_size = 0

    generate_batches = []
    prefill_benchmark_timelist = []
    decoder_benchmark_timelist = []

    while total_req_finished < req_num:
        do_generate = True
        if req_idx < req_num and generate_batch_size < max_batch_size:
            prefill_start = req_idx
            free_block = cache_manager.get_free_block_num()
            total_need_blocks = 0
            total_prefill_token = 0
            prefill_batch_size = 0

            while generate_batch_size + prefill_batch_size < max_batch_size:
                if req_idx >= req_num:
                    break
                cur_need_blocks = req_list[req_idx].need_blocks
                cur_context_len = req_list[req_idx].input_length
                if total_need_blocks + cur_need_blocks > free_block:
                    raise Exception(
                        f"req: {req_idx} out of memory, need block:"
                        + f"{total_need_blocks + cur_need_blocks} is more than free block {free_block}"
                    )
                if cur_context_len > max_prefill_tokens:
                    logger.error(
                        f"req: {req_idx} input length: {cur_context_len} is too long,"
                        + f" max_prefill_tokens: {max_prefill_tokens}"
                    )
                    raise AssertionError
                if total_prefill_token + cur_context_len > max_prefill_tokens:
                    do_generate = False
                    break
                total_need_blocks += cur_need_blocks
                total_prefill_token += cur_context_len
                prefill_batch_size += 1
                req_idx += 1

            if prefill_batch_size > 0:
                batch = Batch(
                    req_list[prefill_start : prefill_start + prefill_batch_size]
                )
                cache_manager.allocate(batch)
                if ENV.benchmark_enable:
                    import time

                    torch.npu.synchronize()
                    prefill_start = time.time()
                    req_finished = generate_token(
                        model,
                        tokenizer,
                        cache_manager,
                        batch,
                        max_out_length,
                        rank,
                        ignore_eos,
                    )
                    torch.npu.synchronize()
                    prefill_end = time.time()
                    prefill_time = prefill_end - prefill_start
                    prefill_benchmark_timelist.append(prefill_time)
                else:
                    req_finished = generate_token(
                        model,
                        tokenizer,
                        cache_manager,
                        batch,
                        max_out_length,
                        rank,
                        ignore_eos,
                    )

                if req_finished != (prefill_batch_size - batch.batch_num):
                    logger.error("batch filter error")
                    raise AssertionError

                if batch.batch_num > 0:
                    generate_batches.append(batch)
                    generate_batch_size += batch.batch_num
                if req_finished > 0:
                    do_generate = False
                    total_req_finished += req_finished

        if do_generate:
            if len(generate_batches) > 1:
                Batch.concatenate(generate_batches)

            if generate_batch_size != generate_batches[0].batch_num:
                logger.error(
                    f"batch concatenate error, expect batchnum: {generate_batch_size},"
                    + f" in fact: {generate_batches[0].batch_num}"
                )
                raise AssertionError

            if ENV.benchmark_enable:
                import time

                torch.npu.synchronize()
                decode_start = time.time()
                req_finished = generate_token(
                    model,
                    tokenizer,
                    cache_manager,
                    generate_batches[0],
                    max_out_length,
                    rank,
                    ignore_eos,
                )
                torch.npu.synchronize()
                decode_end = time.time()
                decode_time = decode_end - decode_start
                decoder_benchmark_timelist.append(decode_time)
            else:
                req_finished = generate_token(
                    model,
                    tokenizer,
                    cache_manager,
                    generate_batches[0],
                    max_out_length,
                    rank,
                    ignore_eos,
                )

            if req_finished != (generate_batch_size - generate_batches[0].batch_num):
                logger.error(f"batch filter error")
                raise AssertionError
            if generate_batch_size > max_generate_batch_size:
                max_generate_batch_size = generate_batch_size
            generate_batch_size = generate_batches[0].batch_num
            if generate_batch_size == 0:
                del generate_batches[0]
            total_req_finished += req_finished

    if rank == 0:
        print("max_generate_batch_size", max_generate_batch_size)
    if ENV.benchmark_enable:
        prefill_time = sum(prefill_benchmark_timelist)
        e2e_time = sum(prefill_benchmark_timelist) + sum(decoder_benchmark_timelist)
        try:
            decode_token_time = sum(decoder_benchmark_timelist) / (max_out_length - 1)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e

        logger.info(
            f"Prefill time: {prefill_time * 1000}ms, "
            f"Decode token time: {decode_token_time * 1000}ms, "
            f"E2E time: {e2e_time * 1000}ms"
        )
        batch_size = len(req_list)
        input_len = req_list[0].input_length
        output_len = max_out_length
        prefill_token_times = ",".join(list(map(str, prefill_benchmark_timelist)))
        decode_token_times = ",".join(list(map(str, decoder_benchmark_timelist)))
        if rank == 0:
            import os

            benchmark_filepath = (
                ENV.benchmark_filepath
                if ENV.benchmark_filepath
                else "./benchmark_result/benchmark.csv"
            )
            benchmark_folder = os.path.dirname(benchmark_filepath)
            if not os.path.exists(benchmark_folder):
                os.makedirs(benchmark_folder)
            stat_data = {
                "batch_size": [batch_size],
                "input_seq_len": [input_len],
                "output_seq_len": [output_len],
                "e2e_time(ms)": [f"{e2e_time * 1000: .2f}"],
                "prefill_time(ms)": [f"{prefill_time * 1000: .2f}"],
                "decoder_token_time(ms)": [f"{decode_token_time * 1000: .2f}"],
                "prefill_count": [len(prefill_benchmark_timelist)],
                "prefill_token_times": [prefill_token_times],
                "decode_token_times": [decode_token_times],
                "max_generate_batch_size": [max_generate_batch_size],
            }
            df = pd.DataFrame(stat_data)
            df.to_csv(benchmark_filepath, index=False)
            logger.info("-------------------performance dumped------------------------")
            df = df.drop("prefill_token_times", axis=1)
            df = df.drop("decode_token_times", axis=1)
            print(df.to_markdown(index=False))


def decode_token(req_list, tokenizer):
    decode_text_list = []
    token_num_list = []
    request_id = 0
    token_num = 0
    for req in req_list:
        out_token = len(req.out_token_list)
        token_tensor = torch.tensor(req.out_token_list, dtype=torch.int64)
        decode_text = tokenizer.decode(token_tensor)
        decode_text_list.append(decode_text)
        token_num += out_token
        token_num_list.append((request_id, token_num))
        request_id += 1
    return decode_text_list, token_num_list

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark online serving throughput with optional KV cache reuse controls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer  # type: ignore


AIOHTTP_RESET_TIMEOUT = aiohttp.ClientTimeout(total=60)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float


class CacheSaltManager:
    """Utility class to control cache salt usage per request."""

    def __init__(self, mode: str, base_salt: Optional[str], seed: int) -> None:
        if mode not in {"reuse", "no-reuse"}:
            raise ValueError(f"Unsupported cache mode: {mode}")
        self.mode = mode
        self.base_salt = base_salt or None
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._counter = 0

    def next(self) -> Optional[str]:
        if self.mode == "reuse":
            return self.base_salt
        # no-reuse mode
        salt_prefix = self.base_salt or "no-reuse"
        salt = f"{salt_prefix}-{self._counter:04d}-{self._rng.getrandbits(32):08x}"
        self._counter += 1
        return salt


async def reset_prefix_cache(reset_url: str) -> None:
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_RESET_TIMEOUT) as session:
            async with session.post(reset_url) as response:
                if response.status != 200:
                    warnings.warn(
                        f"Prefix cache reset endpoint responded with status "
                        f"{response.status}: {await response.text()}",
                        stacklevel=2,
                    )
    except aiohttp.ClientError as exc:
        warnings.warn(
            f"Unable to reset prefix cache via {reset_url}: {exc}", stacklevel=2
        )


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    random.shuffle(dataset)

    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, completion in dataset:
        if len(filtered_dataset) == num_requests:
            break
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids)
            if fixed_output_len is None
            else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    iterator = iter(input_requests)
    for request in iterator:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i, output in enumerate(outputs):
        if output.success:
            output_len = len(
                tokenizer(
                    output.generated_text,
                    add_special_tokens=False,
                ).input_ids
            )
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((output.latency - output.ttft) / (output_len - 1))
            itls.extend(output.itl)
            ttfts.append(output.ttft)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. Please re-check benchmark configuration.",
            stacklevel=2,
        )

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    *,
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    scenario_label: str,
    cache_mode: str,
    cache_salt_manager: Optional[CacheSaltManager],
    reset_prefix_cache_url: Optional[str],
) -> Dict[str, Any]:
    if backend not in ASYNC_REQUEST_FUNCS:
        raise ValueError(f"Unknown backend: {backend}")
    request_func = ASYNC_REQUEST_FUNCS[backend]

    if scenario_label:
        banner = f" KV Cache Mode: {scenario_label} "
        print("\n" + f"{banner:=^60}")

    if reset_prefix_cache_url:
        await reset_prefix_cache(reset_prefix_cache_url)

    def build_extra_body(cache_salt: Optional[str]) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {}
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        if best_of != 1:
            payload["best_of"] = best_of
        if use_beam_search:
            payload["use_beam_search"] = True
        return payload or None

    print("Starting initial single prompt test run...")
    warmup_prompt, warmup_prompt_len, warmup_output_len = input_requests[0]
    warmup_salt: Optional[str] = None
    if cache_salt_manager is not None:
        warmup_salt = cache_salt_manager.next()
    warmup_extra_body = build_extra_body(warmup_salt)
    warmup_input = RequestFuncInput(
        prompt=warmup_prompt,
        prompt_len=warmup_prompt_len,
        output_len=warmup_output_len,
        api_url=api_url,
        model=model_id,
        extra_body=warmup_extra_body,
    )
    warmup_output = await request_func(request_func_input=warmup_input)
    if not warmup_output.success:
        raise ValueError(
            "Initial test run failed - please verify benchmark configuration. "
            f"Error: {warmup_output.error}"
        )
    print("Initial test run completed. Starting main benchmark run...")

    if reset_prefix_cache_url:
        await reset_prefix_cache(reset_prefix_cache_url)
    if cache_salt_manager is not None:
        cache_salt_manager.reset()

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    salts_used: List[Optional[str]] = []

    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        salt: Optional[str] = None
        if cache_salt_manager is not None:
            salt = cache_salt_manager.next()
        extra_body = build_extra_body(salt)
        salts_used.append(salt)

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Time per Output Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "cache_salts": salts_used,
        "kv_cache_mode": cache_mode,
    }
    return result


def save_benchmark_result(
    *,
    args: argparse.Namespace,
    result: Dict[str, Any],
    scenario_mode: str,
    model_id: str,
    tokenizer_id: str,
    results_count: int,
) -> None:
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    backend = args.backend

    result_json: Dict[str, Any] = {
        "date": current_dt,
        "backend": backend,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "best_of": args.best_of,
        "use_beam_search": args.use_beam_search,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate
        if args.request_rate < float("inf")
        else "inf",
        "kv_cache_mode": scenario_mode,
    }

    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                key, value = item.split("=", 1)
                result_json[key.strip()] = value.strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    result_json = {**result_json, **result}

    base_model_id = model_id.split("/")[-1]
    if args.result_filename:
        root, ext = os.path.splitext(args.result_filename)
        if results_count > 1:
            file_name = f"{root}-{scenario_mode}{ext or '.json'}"
        else:
            file_name = args.result_filename
    else:
        file_name = (
            f"{backend}-{scenario_mode}-{args.request_rate}qps-"
            f"{base_model_id}-{current_dt}.json"
        )
    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)
        file_name = os.path.join(args.result_dir, file_name)

    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(result_json, outfile, ensure_ascii=False, indent=2)
    print(f"Saved benchmark result to {file_name}")


def main(args: argparse.Namespace) -> None:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    if args.dataset_name != "sharegpt":
        raise ValueError("Only the ShareGPT dataset is supported in this script.")
    if not args.dataset_path:
        raise ValueError("--dataset-path is required for sharegpt dataset")
    input_requests = sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
    )

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        reset_url = (
            f"{args.base_url}/reset_prefix_cache" if args.reset_prefix_cache else None
        )
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base = f"http://{args.host}:{args.port}"
        reset_url = f"{base}/reset_prefix_cache" if args.reset_prefix_cache else None

    scenario_modes: List[str]
    if args.kv_cache_mode == "both":
        scenario_modes = ["reuse", "no-reuse"]
    else:
        scenario_modes = [args.kv_cache_mode]

    scenario_results: List[Tuple[str, Dict[str, Any]]] = []

    for mode in scenario_modes:
        label = "KV cache reuse enabled" if mode == "reuse" else "No cache reuse"
        manager = CacheSaltManager(mode, args.cache_salt, args.seed)
        result = asyncio.run(
            benchmark(
                backend=args.backend,
                api_url=api_url,
                model_id=args.model,
                tokenizer=tokenizer,
                input_requests=input_requests,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                scenario_label=label,
                cache_mode=mode,
                cache_salt_manager=manager,
                reset_prefix_cache_url=reset_url,
            )
        )
        scenario_results.append((mode, result))

    if args.save_result:
        for mode, result in scenario_results:
            save_benchmark_result(
                args=args,
                result=result,
                scenario_mode=mode,
                model_id=args.model,
                tokenizer_id=tokenizer_id,
                results_count=len(scenario_results),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput with KV cache controls."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL if not using host/port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt"],
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--sharegpt-output-len", type=int, default=None)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second. Use inf to send all at once.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Metadata key-value pairs to embed in result JSON.",
    )
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument(
        "--kv-cache-mode",
        type=str,
        choices=["reuse", "no-reuse", "both"],
        default="reuse",
        help="Control whether KV cache reuse is enabled, disabled, or benchmarked in both modes.",
    )
    parser.add_argument(
        "--cache-salt",
        type=str,
        default=None,
        help="Optional cache salt to use when KV cache reuse is enabled."
        " When not set, requests share the default cache namespace.",
    )
    parser.add_argument(
        "--reset-prefix-cache",
        action="store_true",
        help="Attempt to call the server's /reset_prefix_cache endpoint before measurements.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)

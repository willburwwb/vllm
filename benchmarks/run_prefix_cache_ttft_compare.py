#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TTFT with and without prefix caching under long-context workloads."""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:  # pragma: no cover - fallback for standalone benchmark use
    from benchmarks.backend_request_func import get_tokenizer  # type: ignore

from benchmark_prefix_caching import (  # type: ignore
    Request,
    repeat_and_sort_requests,
    sample_requests_from_dataset,
    sample_requests_from_random,
)


@dataclass
class ScenarioResult:
    name: str
    wall_time: float
    ttfts: list[float]
    cached_tokens: list[int]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_vals[int(k)]
    frac = k - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return statistics.fmean(values)


def create_argument_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark TTFT (time-to-first-token) for long prompts with and "
            "without prefix caching enabled."
        )
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--input-length-range", type=str, required=True)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort prompts by input length (helps determinism).",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help="Exclude detokenization time from measurements.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup iterations before measurement (0 disables).",
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser


def build_requests(
    args, tokenizer
) -> tuple[list[Request], list[str]]:  # type: ignore[return-type]
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    if args.dataset_path:
        if args.prefix_len > 0:
            raise ValueError("prefix-len cannot be used with dataset-path")
        requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )

    prompts = repeat_and_sort_requests(
        requests, repeat_count=args.repeat_count, sort=args.sort
    )
    return requests, prompts


def summarize(result: ScenarioResult) -> None:
    ttfts = result.ttfts
    cached_tokens = result.cached_tokens
    mean_ttft = average(ttfts)
    print(f"\n[{result.name}] Wall time: {result.wall_time:.3f}s")
    if ttfts:
        print(
            "  TTFT mean/p50/p95/p99: "
            f"{mean_ttft:.4f}s / "
            f"{percentile(ttfts, 50):.4f}s / "
            f"{percentile(ttfts, 95):.4f}s / "
            f"{percentile(ttfts, 99):.4f}s"
        )
    else:
        print("  No TTFT samples collected (metrics missing).")
    if cached_tokens:
        avg_cached = average(cached_tokens)
        print(
            "  Avg cached tokens per request: "
            f"{avg_cached:.1f} (max {max(cached_tokens)}, min {min(cached_tokens)})"
        )


def run_scenario(
    name: str,
    enable_prefix_caching: bool,
    engine_kwargs: dict,
    prompts: list[str],
    sampling_params: SamplingParams,
    warmup: int,
) -> ScenarioResult:
    engine_kwargs = dict(engine_kwargs)
    engine_kwargs["enable_prefix_caching"] = enable_prefix_caching
    llm = LLM(**engine_kwargs)
    try:
        if warmup > 0:
            warmup_prompts = prompts[: max(1, min(warmup, len(prompts)))]
            llm.generate(warmup_prompts, sampling_params=sampling_params)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        wall_time = time.perf_counter() - start

        ttfts: list[float] = []
        cached_tokens: list[int] = []
        for out in outputs:
            metrics = getattr(out, "metrics", None)
            if metrics and metrics.first_token_time is not None:
                ttft = metrics.first_token_time - metrics.arrival_time
                ttfts.append(ttft)
            if out.num_cached_tokens is not None:
                cached_tokens.append(out.num_cached_tokens)

        return ScenarioResult(name, wall_time, ttfts, cached_tokens)
    finally:
        llm.llm_engine.shutdown()


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    _, prompts = build_requests(args, tokenizer)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    engine_args = EngineArgs.from_cli_args(args)
    engine_kwargs = dataclasses.asdict(engine_args)

    with_cache = run_scenario(
        name="Prefix cache ON",
        enable_prefix_caching=True,
        engine_kwargs=engine_kwargs,
        prompts=prompts,
        sampling_params=sampling_params,
        warmup=args.warmup,
    )
    summarize(with_cache)

    without_cache = run_scenario(
        name="Prefix cache OFF",
        enable_prefix_caching=False,
        engine_kwargs=engine_kwargs,
        prompts=prompts,
        sampling_params=sampling_params,
        warmup=args.warmup,
    )
    summarize(without_cache)


if __name__ == "__main__":
    main()

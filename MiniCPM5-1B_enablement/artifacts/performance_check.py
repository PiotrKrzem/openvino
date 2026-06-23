#!/usr/bin/env python3
"""
performance_check.py -- MiniCPM5-1B OpenVINO performance measurement (Phase 4).

Measures latency / throughput of the stateful MiniCPM IR and writes
perf_results.json for CI artifact upload. The model is a causal LM, so we
report both first-token (prefill) latency and the per-iteration generate
latency for a fixed number of new tokens.

Usage:
    python performance_check.py --model ./ov_minicpm [--device CPU]
        [--niter 20] [--nwarmup 3] [--max-new-tokens 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_PROMPT = "OpenVINO is an open-source toolkit for optimizing and deploying AI inference."


def run_performance_check(model_path: str, device: str, niter: int, nwarmup: int,
                          max_new_tokens: int, prompt: str, trust_remote_code: bool) -> dict:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = OVModelForCausalLM.from_pretrained(
        model_path, device=device, trust_remote_code=trust_remote_code,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                      do_sample=False, num_beams=1)

    # Warmup
    for _ in range(nwarmup):
        model.generate(**inputs, **gen_kwargs)

    latencies = []
    for _ in range(niter):
        t0 = time.perf_counter()
        model.generate(**inputs, **gen_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.asarray(latencies)
    tokens_per_iter = max_new_tokens
    total_tokens = tokens_per_iter * niter
    total_time_s = lat.sum() / 1000.0

    return {
        "model": str(model_path),
        "device": device,
        "niter": niter,
        "nwarmup": nwarmup,
        "max_new_tokens": max_new_tokens,
        "gen_latency_median_ms": round(float(np.median(lat)), 3),
        "gen_latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "gen_latency_mean_ms": round(float(np.mean(lat)), 3),
        "per_token_latency_ms": round(float(np.mean(lat) / tokens_per_iter), 3),
        "throughput_tok_s": round(total_tokens / total_time_s, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR directory")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--niter", type=int, default=20)
    parser.add_argument("--nwarmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    try:
        results = run_performance_check(args.model, args.device, args.niter, args.nwarmup,
                                        args.max_new_tokens, args.prompt, args.trust_remote_code)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(2)

    print(json.dumps(results, indent=2))
    Path("perf_results.json").write_text(json.dumps(results, indent=2))
    print("\n[SAVED] perf_results.json", file=sys.stderr)
    sys.exit(0)

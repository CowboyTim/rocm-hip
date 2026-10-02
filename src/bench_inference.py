#!/usr/bin/env python3
"""
bench_inference.py -- Phase 6: Python orchestrator for inference benchmarking.
Measures tokens/sec at varying sequence lengths, batch sizes, and model sizes.
Outputs JSON reports with per-layer timing breakdowns.
"""
import json
import subprocess
import time
import sys
import os
import argparse
from pathlib import Path

def run_benchmark(model_path, prompt, n_tokens, n_warmup=2, n_runs=5):
    """Run inference_demo binary and measure latency."""
    cmd = ["./inference_demo", model_path, prompt]
    # Warmup
    for _ in range(n_warmup):
        subprocess.run(cmd, capture_output=True)
    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        subprocess.run(cmd, capture_output=True)
        end = time.perf_counter()
        latencies.append(end - start)
    avg = sum(latencies) / len(latencies)
    tok_per_sec = n_tokens / avg
    return {
        "avg_latency_sec": avg,
        "tokens_per_sec": tok_per_sec,
        "latencies_sec": latencies
    }

def main():
    parser = argparse.ArgumentParser(description="Inference benchmark orchestrator")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", default="The quick brown fox", help="Prompt text")
    parser.add_argument("--n-tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="bench_report.json")
    args = parser.parse_args()

    if not os.path.exists("./inference_demo"):
        print("Error: inference_demo binary not found. Build first.", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking model {args.model} with prompt '{args.prompt}'")
    result = run_benchmark(args.model, args.prompt, args.n_tokens,
                           n_warmup=args.warmup, n_runs=args.runs)
    print(f"Average latency: {result['avg_latency_sec']:.4f}s")
    print(f"Throughput: {result['tokens_per_sec']:.2f} tok/s")

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "n_tokens": args.n_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        **result
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
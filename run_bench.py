#!/usr/bin/env python3
"""
ROCm HIP FP4 Matmul Benchmark Orchestrator

Runs benchmarks comparing different FP4 quantization methods on ROCm GPUs.
Tests: FP4->FP16 WMMA + LDS LUT, DP4A hardware intrinsics, scalar fallback.
"""

import subprocess
import sys
import os
import argparse
import json
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

# Configuration
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
HIPCC = f"{ROCM_PATH}/bin/hipcc"
CLANG = f"{ROCM_PATH}/lib/llvm/bin/clang++"
ARCH = "gfx1151"
FP4_CODEBOOK = np.array([0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10], dtype=np.int8)

class BenchmarkResult:
    def __init__(self, name, latency_ms, gflops, mismatch_pct, max_diff, status):
        self.name = name
        self.latency_ms = latency_ms
        self.gflops = gflops
        self.mismatch_pct = mismatch_pct
        self.max_diff = max_diff
        self.status = status  # 'pass' or 'fail'

def compile_kernel(hip_file, output_co):
    """Compile HIP kernel to .co object file using Makefile rules."""
    cmd = ["make", output_co]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return True

def run_benchmark(M, N, K, runs, warmup, scale):
    """Run the wmma_bench binary and parse results."""
    cmd = ["./wmma_bench", "-M", str(M), "-N", str(N), "-K", str(K),
           "-runs", str(runs), "-warmup", str(warmup), "-scale", str(scale)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout

def parse_benchmark_output(stdout):
    """Parse benchmark output to extract all kernel results."""
    results = []
    lines = stdout.split('\n')
    
    results_map = {}  # kernel_name -> {latency, gflops, mismatch_pct, max_diff}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for results section header
        if "Results for" in line:
            full_name = line.split("Results for")[1].strip().rstrip(':').strip()
            
            # Extract latency and throughput from next lines
            latency = None
            gflops = None
            for j in range(i+1, min(i+5, len(lines))):
                if "Avg latency:" in lines[j]:
                    parts = lines[j].split(":")
                    if len(parts) >= 2:
                        latency = float(parts[1].strip().split()[0])
                if "Throughput:" in lines[j]:
                    parts = lines[j].split(":")
                    if len(parts) >= 2:
                        gflops = float(parts[1].strip().split()[0])
            
            if latency:
                results_map[full_name] = {
                    'latency': latency,
                    'gflops': gflops if gflops else 0.0,
                    'mismatch_pct': 100.0,
                    'max_diff': 0.0,
                    'status': 'unknown'
                }
            i += 1
            continue
        
        # Look for verification lines (after "Verification vs CPU Reference")
        if "Mismatch:" in line:
            import re
            # Extract kernel short name from line like "Scalar LUT:         Mismatch:..."
            for name in results_map.keys():
                short_name = name.split()[0]
                if short_name in line or name in line:
                    match = re.search(r'Mismatch:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)', line)
                    if match:
                        results_map[name]['mismatch_pct'] = float(match.group(3))
                    
                    # Look for Max diff in next line
                    if i+1 < len(lines) and "Max diff:" in lines[i+1]:
                        parts = lines[i+1].split(":")
                        if len(parts) >= 2:
                            try:
                                results_map[name]['max_diff'] = float(parts[1].strip())
                            except ValueError:
                                pass
                    
                    # Determine pass/fail
                    mismatch = results_map[name]['mismatch_pct']
                    results_map[name]['status'] = 'pass' if mismatch < 5.0 else 'fail'
                    break
        
        i += 1
    
    # Convert to list
    for name, data in results_map.items():
        results.append(BenchmarkResult(
            name, data['latency'], data['gflops'],
            data['mismatch_pct'], data['max_diff'], data['status']
        ))
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="ROCm HIP FP4 Matmul Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bench.py -M 256 -N 256 -K 256 --runs 20
  python run_bench.py --small  # Quick test with 16x16 matrices
  python run_bench.py --json   # Output as JSON
        """
    )
    parser.add_argument("-M", type=int, default=256, help="Matrix M dimension")
    parser.add_argument("-N", type=int, default=256, help="Matrix N dimension")
    parser.add_argument("-K", type=int, default=256, help="Matrix K dimension")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup runs")
    parser.add_argument("--scale", type=float, default=1.0, help="FP4 scale factor")
    parser.add_argument("--small", action="store_true", help="Quick test: 16x16 matrices")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    if args.small:
        M, N, K = 16, 16, 16
        runs, warmup = 5, 2
    else:
        M, N, K = args.M, args.N, args.K
        runs, warmup = args.runs, args.warmup
    
    print("=" * 78)
    print("  ROCm HIP FP4 Matmul Benchmark")
    print("=" * 78)
    print(f"  Matrix size:  {M} x {K} x {N}")
    print(f"  FP4 format:   ROCmFP4 Codebook10")
    print(f"  Values:       0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10")
    print(f"  Scale:        {args.scale}")
    print(f"  Runs:         {runs} (after {warmup} warmup)")
    print(f"  Total MAC:    {2.0 * M * K * N / 1e9:.1f} billion")
    print("=" * 78)
    
    # Compile kernels
    print("\nCompiling HIP kernels...")
    kernels = [
        "wmma_bench_fp4_dp4a",
        "wmma_bench_fp4_dp4a_hardware", 
        "wmma_bench_fp4_native",
        "wmma_bench_fp4_lut",
        "wmma_bench_fp4_lut_fixed",
        "wmma_bench_fp4_dp4a_int8"
    ]
    
    for kernel in kernels:
        co_file = f"{kernel}.co"
        if compile_kernel(f"{kernel}.hip", co_file):
            print(f"  OK: {kernel}")
        else:
            print(f"  FAIL: {kernel}")
            return 1
    
    # Build benchmark binary
    print("\nBuilding wmma_bench binary...")
    result = subprocess.run(["make", "wmma_bench"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  FAIL: Build failed")
        print(result.stderr)
        return 1
    print("  OK: wmma_bench")
    
    # Run benchmarks
    print("\n" + "=" * 78)
    print("  Running Benchmarks")
    print("=" * 78)
    
    success, stdout = run_benchmark(M, N, K, runs, warmup, args.scale)
    
    if not success:
        print("  FAIL: Benchmark crashed")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
        return 1
    
    results = parse_benchmark_output(stdout)
    
    # Summary
    print("\n" + "=" * 78)
    print("  Summary")
    print("=" * 78)
    
    if not results:
        print("  No results parsed from benchmark output")
        return 1
    
    # Sort by speed (lower latency first)
    results_sorted = sorted(results, key=lambda r: r.latency_ms)
    
    print(f"\n  {'Method':<45s} {'Latency':>10s} {'GFLOPS':>10s} {'Mismatch':>10s} {'Status':>8s}")
    print("  " + "-" * 70)
    
    for r in results_sorted:
        print(f"  {r.name:<45s} {r.latency_ms:>9.3f}ms {r.gflops:>9.1f} {r.mismatch_pct:>9.2f}% {r.status:>8s}")
    
    # Speedups
    if len(results_sorted) >= 2:
        print("\n  --- Speedups (relative to slowest) ---")
        slowest = results_sorted[-1].latency_ms
        for r in results_sorted:
            speedup = slowest / r.latency_ms
            print(f"  {r.name:<45s} {speedup:8.2f}x")
    
    # JSON output if requested
    if args.json:
        output = {
            "config": {
                "M": M, "N": N, "K": K,
                "scale": args.scale, "runs": runs
            },
            "results": [
                {
                    "name": r.name,
                    "latency_ms": r.latency_ms,
                    "gflops": r.gflops,
                    "mismatch_pct": r.mismatch_pct,
                    "max_diff": r.max_diff,
                    "status": r.status
                }
                for r in results_sorted
            ]
        }
        print(f"\n  JSON output:")
        print(json.dumps(output, indent=2))
    
    print("\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

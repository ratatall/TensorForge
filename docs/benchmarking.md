# Reproducible benchmarks

## Environment and evidence

Re-measured after the audit and O2 extension on 2026-09-06 UTC: macOS 15.5 / Darwin 24.5.0 arm64, Apple Clang 17.0.0.17000013, LLVM 23.1.0, CMake 4.4.3, Release, host sanitizers off. LLVM reports host target CPU `apple-m4`. No affinity, frequency locking, thermal control, or exclusive-machine reservation was used.

```bash
scripts/build.sh Release build-audit-final-release
scripts/test.sh build-audit-final-release
python3 scripts/benchmark.py --binary build-audit-final-release/tensorforge --iterations 101 --output results/audit-final
python3 scripts/benchmark.py --binary build-audit-final-release/tensorforge --iterations 101 --output results/audit-repeat
```

[Primary CSV](../benchmarks/results.csv) and [repeat CSV](../benchmarks/repeat-results.csv) each contain 45 rows: nine workloads, the interpreter, and all four combinations of TensorForge passes and LLVM middle-end optimization. Every row records verified output. These replace the initial project's measurements because target attributes and code generation changed. The runner regenerates all workload sources and rejects incomplete/unverified CSV output.

## Primary execution medians

All times are microseconds; TF off/on selects custom passes and none/O2 selects LLVM middle-end optimization.

| Workload | N | Interpreter µs | TF off / none | TF on / none | TF off / O2 | TF on / O2 |
|---|---:|---:|---:|---:|---:|---:|
| single | 256 | 0.524 | 0.067 | 0.067 | 0.012 | 0.012 |
| chain | 256 | 1.766 | 0.284 | 0.078 | 0.041 | 0.020 |
| broadcast | 256 | 1.330 | 0.213 | 0.069 | 0.029 | 0.016 |
| single | 16,384 | 25.885 | 3.956 | 4.010 | 0.722 | 0.722 |
| chain | 16,384 | 92.854 | 16.693 | 5.095 | 5.459 | 1.291 |
| broadcast | 16,384 | 71.708 | 22.219 | 4.398 | 3.512 | 1.192 |
| single | 262,144 | 582.417 | 75.969 | 68.854 | 19.456 | 18.844 |
| chain | 262,144 | 1874.208 | 272.375 | 81.906 | 84.291 | 22.161 |
| broadcast | 262,144 | 1287.375 | 198.167 | 67.979 | 52.114 | 19.383 |

Measured example, N=262,144, seed 42, 101 samples (macOS arm64, LLVM target `apple-m4`):

| TensorForge passes | LLVM middle end | Broadcast/ReLU median | Scratch |
|---|---|---:|---:|
| off | none | 198.167 µs | 2 MiB |
| on | none | 67.979 µs | 0 |
| off | O2 | 52.114 µs | 2 MiB planned |
| on | O2 | 19.383 µs | 0 |

With LLVM middle-end optimization held at none, custom fusion reduced median execution time by **65.7% (2.92×)** against **TensorForge’s unfused JIT path**. The repeat measured 2.98×. LLVM O2 adds a separate benefit, with fused medians of 19.383 and 22.745 µs in the two runs. The controls and variability below limit how broadly these local microbenchmark results can be interpreted.

## Variability and controls

The primary broadcast/none unfused p10–p90 was 196.625–213.167 µs; fused was 65.990–90.948 µs. The primary fused/O2 interval was 18.578–31.036 µs. The repeat fused/O2 median was 22.745 µs versus 19.383 µs initially, so its exact speedup is less stable than the custom-fusion comparison. Both full runs are retained; no best-of-two selection is used.

The single-add control lowers to the same one loop and zero scratch with project passes off/on. At N=262,144 it measured 75.969 versus 68.854 µs in the primary run, but 67.667 versus 68.333 µs in the repeat. This illustrates noise/order effects; no project-pass gain is claimed for a single operation. LLVM O2 vectorizes the inspected fused broadcast loop, supported by [IR and assembly evidence](llvm-optimization.md), not inferred solely from timing.

## Workloads and structure

| Name | Expression | Unfused lowered loops | Fused lowered loops | Unfused scratch floats |
|---|---|---:|---:|---:|
| single | `A + B` | 1 | 1 | 0 |
| chain | `relu((A + B) * B + A)` | 4 | 1 | 3N |
| broadcast | `relu(A * 2.0 + B)` | 3 | 1 | 2N |

All fused workloads require zero tensor scratch. At N=262,144 the broadcast chain saves 2,097,152 bytes of planned scratch. Inputs and output still occupy memory. `lowered_loops` is TensorForge's pre-LLVM count; `llvm_loops` is actual post-pipeline LoopInfo count. Vector/remainder loops can change the latter. `vector_instructions` counts vector-typed IR results, not retired machine instructions. Scratch bytes are the host allocation contract, not measured traffic or an LLVM liveness estimate.

All modes use backend `CodeGenOptLevel::Default`. O2 runs the standard PassBuilder middle-end pipeline; none runs no such pipeline. Neither enables fast-math. The primary fusion claim compares project off/on with LLVM none; O2 gains are reported separately.

## Timing boundaries

`benchmarks/benchmark.cpp` enforces a Release build and uses `steady_clock`. Inputs are deterministic: `mt19937(seed)` followed by a specified integer mapping into [-4, 4]. The default seed is 42 and the saved runs use 101 timing samples per mode.

Each of the five modes receives five warmup calls. A calibration step doubles calls per sample until a batch lasts at least 200 µs, capped at 65,536 calls. Each sample is divided by batch size, and the median of those per-call means is reported. Thus `--iterations` counts samples, not total kernel invocations. The CSV records actual batch sizes, p10 and p90 sample quantiles, both optimization dimensions, target CPU, sanitizer state, and verification status. Quantiles summarize within-run sample variation, not confidence intervals. Warmup and calibration are excluded from reported samples.

Parsing and semantic analysis happen before benchmark timing. Each compile measurement includes IR validation/copy, native target/JIT setup, IR generation, verification, and symbol lookup/native compilation. Project-on measurements include custom passes; LLVM-O2 measurements include the standard middle-end pipeline. All paths copy and validate the original IR. These are single latency samples, not medians. The first JIT also pays process initialization, so **do not claim faster optimized compilation from these numbers**.

JIT execution samples reuse already allocated inputs/output/scratch and call `invoke` through an opaque native function pointer. They include function-call and checksum overhead but exclude allocation, runtime input validation, and JIT compilation. Each call consumes first/last output values in an accumulated checksum. Every output element is compared with the interpreter before timing, and JIT output is checked again after each measurement. The native function call remains externally observable to the host compiler.

Interpreter samples include IR/input validation, input copies, and per-operation vector allocations. They serve as a simple reference execution cost, not a claim that JIT arithmetic alone is that many times faster. Interpreter compile/loop/vector/scratch CSV fields are zero sentinels for “not a JIT lowering statistic,” not assertions that the interpreter allocates no memory or runs no loops.

## Limits and reproduction advice

Modes run in fixed order (interpreter, unfused/none, fused/none, unfused/O2, fused/O2), using the same data repeatedly; this favors warm caches. Results are not cold-cache bandwidth measurements. Timer overhead matters for tiny kernels even with batching. Two complete runs provide a repeat check, with visible timing variation; they do not establish confidence intervals or portability. Use several independent runs under comparable load before reporting a new performance result.

If a new workload regresses, inspect the custom schedule, LLVM loop count, scratch count, and Release configuration. A single operation has no intermediate traffic to remove. A large expression DAG may incur register pressure. Keep unfavorable results visible and distinguish TensorForge passes from the separately selected LLVM optimization pipeline.

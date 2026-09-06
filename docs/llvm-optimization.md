# Independent LLVM middle-end optimization

TensorForge now exposes two independent switches:

| TensorForge passes | LLVM middle end | CLI options |
|---|---|---|
| off | none | `--llvm-opt=none` |
| on | none | `--opt --llvm-opt=none` |
| off | O2 | `--llvm-opt=O2` |
| on | O2 | `--opt --llvm-opt=O2` |

Ordinary `run` and `emit-llvm` preserve their defaults: project passes off and LLVM middle end none. `--verify` checks the displayed result and all four compiled configurations against the unoptimized interpreter. `benchmark` always reports all four plus the interpreter and accepts neither selector. `--interpret` requires project passes off and LLVM none, though adding `--verify` then constructs all JIT configurations to compare them.

## Implementation and legality

`Executable(module, LLVMOptimization)` in `src/codegen.cpp` generates and verifies the original function/module. For O2 it constructs LLVM's Loop, Function, CGSCC, and Module analysis managers, registers them with `PassBuilder`, cross-registers analysis proxies, and runs `buildPerModuleDefaultPipeline(OptimizationLevel::O2)`. The PassBuilder receives the same host TargetMachine used to identify CPU/features. Function and module verification run again before printing or ORC submission.

Both modes retain backend `CodeGenOptLevel::Default`. `none` means no LLVM middle-end pipeline, not “LLVM performs no code generation optimizations.” Both use builder constant folding, instruction selection, register allocation, and normal machine-level transformations. Both now carry matching host CPU/feature attributes and output/scratch `noalias` attributes derived from the existing disjoint-buffer ABI. Inputs may still alias one another. All performance data was regenerated after these attributes were added.

No fast-math, reassociation, no-NaN, no-signed-zero, or contraction flags are enabled. O2 transformations must preserve the documented ReLU policy and binary32 semantics. Tests cover NaNs, infinities, signed zeros, broadcasting, constants, dead code, shared DAGs, and fixed/generated expressions in all four configurations. A structural test checks that O2 removes `x * 1` from emitted LLVM IR without relying on a particular CPU's vector width. Modules are verified before and after the pipeline.

ORC's current-process symbol generator resolves libc calls such as memcpy if standard optimization/backend lowering introduces them. The owning LLJIT still outlives every call; this is native in-process execution, not a remote executor.

## What the statistics mean

`lowered_loops` counts the loops TensorForge constructs before LLVM passes. `llvm_loops` uses LLVM LoopInfo after the selected pipeline; vector/remainder loops can change this count. `vector_instructions` counts vector-typed result instructions in the post-pipeline IR, including vector loads and compares, not only floating-point arithmetic. These are structural counts, not executed instruction counts or hardware counters.

`scratch_bytes` is the host allocation contract planned by TensorForge lowering. O2 may eliminate some accesses, but the host still reserves that planned buffer. It must not be advertised as measured memory traffic or post-O2 peak memory usage.

## Reproduce the assembly evidence

First run the benchmark source generator or create the same two-input program with N=262144:

```bash
python3 scripts/benchmark.py --binary build/tensorforge --output results
build/tensorforge emit-llvm results/broadcast_262144.tf --opt > results/fused-none.ll
build/tensorforge emit-llvm results/broadcast_262144.tf --opt --llvm-opt=O2 > results/fused-O2.ll
# Use LLVM 23 tools on PATH (opt-23/llc-23 on a versioned Linux install).
opt -passes=verify -disable-output results/fused-none.ll
opt -passes=verify -disable-output results/fused-O2.ll
llc -O2 results/fused-none.ll -o results/fused-none.s
llc -O2 results/fused-O2.ll -o results/fused-O2.s
```

On Homebrew, use `"$(brew --prefix llvm)/bin/opt"` and `"$(brew --prefix llvm)/bin/llc"` if they are not on PATH. Both `llc` commands use the same backend optimization setting. These commands recompile the emitted IR; they are not a disassembly of LLJIT's exact in-memory bytes.

The checked-in [LLVM none IR](../benchmarks/codegen/fused-none.ll), [O2 IR](../benchmarks/codegen/fused-O2.ll), [none assembly](../benchmarks/codegen/fused-none.s), and [O2 assembly](../benchmarks/codegen/fused-O2.s) are intentional, target-specific inspection artifacts generated with LLVM 23.1.0 for the detected `apple-m4` target. The none path has scalar arithmetic. The O2 path contains `<4 x float>` operations and assembly such as:

```asm
fadd.4s  v0, v0, v0
fadd.4s  v0, v0, v4
fcmgt.4s v4, v0, #0.0
and.16b  v0, v4, v0
```

Four vector groups process 16 elements per loop iteration. The vector compare/mask implements the same ReLU NaN/zero policy. This supports a vectorization claim for this inspected workload/target; it does not establish SIMD on every expression or CPU. Read [benchmark results](benchmarking.md) to separate fusion's contribution from LLVM O2, including variability between independent runs.

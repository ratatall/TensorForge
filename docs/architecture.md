# Architecture and runtime

TensorForge separates tensor meaning from low-level execution. The public model is in three small headers; implementation files group closely related components instead of creating an empty directory for every conceptual class.

1. `Source::fail` and `lex` in `src/frontend.cpp` track byte offsets and one-based line/column locations. Tokens own their spellings, so they do not depend on temporary views.
2. `Parser` owns a token vector and builds a variant-based AST. `unique_ptr` owns expression children; declarations and return retain locations. Recursive descent encodes precedence directly.
3. `Analyzer` in `src/ir.cpp` checks declaration order, one namespace, and shapes while constructing typed IR. This combines semantic analysis and IR building in one walk. No partially analyzed module escapes on failure.
4. IR operations are stored in definition order. Operand indices always refer backward; every value has a type. The return is a module field rather than a value-producing operation. `validateIR` checks the contract before execution and after each pass.
5. `PassManager` runs constant folding, DCE, and fusion scheduling. The CLI retains original IR for the interpreter and copies it before optimizing.
6. `src/loop_ir.cpp` lowers verified Tensor IR to an explicit ordered schedule of scalar computations, elementwise loops, copies, fused loops, and nested reductions. Its verifier reconstructs the legal schedule from Tensor IR and rejects incomplete or reordered plans.
7. `Generator` in `src/codegen.cpp` consumes Loop IR, emits a function, verifies it and its module, optionally runs standard LLVM O2, verifies again, and hands ownership to ORC. `Executable` owns LLJIT for at least as long as its callable address is used.
8. `src/main.cpp` converts located diagnostics and internal/runtime exceptions into nonzero exit codes. LLVM `Error` and `Expected` are always consumed and converted to readable exceptions at the boundary.

## Runtime ABI

```cpp
extern "C" void tensorforge_run(
    const float* const* inputs,
    float* output,
    float* scratch);
```

The generated LLVM signature is `void (ptr, ptr, ptr)`. C++ constness is a host-side contract; LLVM uses opaque pointers. TensorForge targets a 64-bit native host and emits 64-bit loop indices.

`inputs` points to an array of pointers in source declaration order, including unused inputs. Scalar inputs point to one float. `output` has the returned type's element count, including one element for scalar returns. `scratch` contains the number of floats reported by `Executable::stats()`. Zero-length unused inputs/scratch pointer arrays may be null; the generated code does not dereference them when unused.

Inputs may alias each other because the kernel only reads them. Output and scratch must be separate writable buffers, disjoint from every input and each other. `Executable::run` enforces sizes and owns these allocations. `invoke` is a prepared, unchecked entry point for the benchmark; callers must meet these preconditions and keep every buffer alive. Concurrent calls need independent output/scratch buffers. A moved-from executable must not be invoked.

The third pointer is a deliberate adjustment to the suggested two-pointer ABI. It prevents large stack arrays or calls to allocation routines inside generated functions. Both JIT benchmark paths exclude host allocation; the unfused path still pays actual intermediate stores and loads.

## Two lowerings

Unfused: assign an output/scratch pointer to each live tensor computation, generate one flattened row-major loop per tensor operation, and load predecessor tensors from their buffers. Broadcast indices are derived from the consumer's flat index without materializing expanded tensors. The return computation writes directly to output; intermediate values have distinct scratch slices. A directly returned tensor input gets a copy loop. Scalars remain LLVM SSA values and are computed once, not per lane.

Fused: consume the explicit topological schedule in one output loop when all tensor computations have the returned shape. Values produced during the current lane live in an LLVM SSA cache. A shared DAG node is computed once per lane even if several users read it. Scalar computations precede the loop. Broadcast input loads are indexed and cached per output lane. No tensor intermediates are materialized. Mixed-shape intermediate graphs remain on the correct unfused path rather than duplicating smaller computations.

Reduction: generate one loop over result elements and a nested loop over the selected input axis. Row-major index reconstruction maps each result coordinate and reduction coordinate back to the source tensor without transposing or expanding it. Rank-one reductions produce an LLVM scalar that later scalar or tensor operations can consume. The current fusion pass remains conservative when a reduction is live, so elementwise producers are materialized before reduction rather than duplicated or illegally reordered.

All paths use the same host target and LLVM backend `CodeGenOptLevel::Default`. The LLVM middle-end selector defaults to `none`; `O2` invokes the standard PassBuilder per-module pipeline with the host TargetMachine and registered analysis managers. LLVM performs instruction selection/register allocation and builder-level constant folding in either case. TensorForge owns the measured change in loop count and scratch traffic; it does not claim every machine-level optimization as its own.

## Ownership and failures

The AST is move-only. IR uses owned vectors and strings; copying IR creates an independent optimization candidate. LLVM context/module unique pointers move into `ThreadSafeModule`; LLJIT owns compiled resources. `ExecutorAddr::toPtr<Kernel>()` provides the supported conversion for in-process execution. Function lookup forces compilation before benchmark execution timing begins. A current-process symbol generator resolves libc helpers (such as memcpy) that O2 may introduce. Output and scratch carry LLVM `noalias` attributes matching the existing ABI contract. Target CPU/features are attached to the generated function; emitted IR and JIT use the same target settings.

One process-wide `once_flag` is required for LLVM native target registration. There is no mutable compiler session singleton. Exceptions are used consistently: `Diagnostic` for source errors, `logic_error` for internal invariant violations, and runtime errors for LLVM or I/O failures. This keeps a small compiler readable without mixing three unrelated error propagation styles.

Input generation, interpreter materialization, and JIT scratch planning now preflight separate 1 GiB budgets with checked byte arithmetic. The IR verifier rejects unknown live opcodes and oversized input/result types before lowering. These limits protect against excessive individual working sets, not every possible source-level denial of service.

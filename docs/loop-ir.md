# Loop IR

Loop IR is the explicit boundary between tensor semantics and LLVM construction. Tensor IR describes values and shapes; Loop IR describes the ordered computations needed to produce those values.

```text
loop_module {
  loop0 elementwise [%1] over tensor<2x4xf32>
  loop1 reduction [%2] over tensor<2xf32> axis=1 extent=4
}
```

Each entry has one of five kinds:

| Kind | Meaning |
|---|---|
| `scalar` | Evaluate one scalar Tensor IR operation in LLVM SSA |
| `elementwise` | Run one flattened row-major output loop |
| `reduction` | Run an output loop with a nested reduction-axis loop |
| `copy` | Copy a directly returned tensor input to the output ABI buffer |
| `fused_elementwise` | Evaluate a verified Tensor IR fusion schedule in one output loop |

The lowering preserves Tensor IR definition order. Constants and scalar inputs are initialized directly and therefore need no schedule entry. A reduction records its result iteration type, source axis, and reduction extent. LLVM lowering reconstructs row-major source indices from the output index and reduction coordinate.

`validateLoopIR` rebuilds the canonical schedule from verified Tensor IR and compares it with the supplied Loop IR. Missing, reordered, or altered entries are rejected before LLVM construction. This deliberately small representation makes scheduling visible and testable without turning it into a general control-flow IR.

Inspect it with:

```bash
build/tensorforge dump-loop-ir examples/reduction.tf --opt
```

Elementwise fusion is currently conservative around reductions. When a reduction is live, its elementwise producer is materialized first and the reduction reads that buffer. A future producer-reduction fusion pass can change the Loop IR schedule while preserving the Tensor IR and interpreter oracle.

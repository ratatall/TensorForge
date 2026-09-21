# Tensor IR

A `Module` owns `operations`, the stable `inputs` ABI table, a returned `ValueId`, and an optional `fusedRegion` schedule. An `Operation` holds opcode, result type, operands, source location, optional input/constant metadata, and a liveness bit. Value IDs are vector indices, not pointers.

| Opcode | Operands | Result |
|---|---|---|
| `input` | none; ABI argument index | declared scalar/tensor |
| `constant` | none; binary32 value | scalar |
| `add` | left, right | broadcast-compatible type |
| `multiply` | left, right | broadcast-compatible type |
| `relu` | argument | same type |
| `reduce_sum` | argument, compile-time axis attribute | input type with that axis removed |

A `let` is a symbol-table alias for an existing value, not a copy operation. There are no mutation, blocks, branches, or PHI nodes at this level. It is SSA-like because each ID is defined once and every use refers to a prior definition. LLVM PHI nodes appear later to implement loops.

## Rewriting and verification

Constant folding changes a computation into a constant at the same ID and clears its operands. DCE marks unreachable operations dead; it does not erase vector entries or renumber users. Thus IDs can have gaps in a dump. The input signature is never compacted, even when an input operation becomes dead.

The verifier checks supported live opcodes, bounded ranks and element counts, return validity, live backward references, operand arity, input signature types, broadcast-derived result shapes, and the fusion schedule. A scheduled region must contain all live non-input tensor computations in definition order, all with the returned shape, and end at the return value. Invalid IR is an internal error, not user syntax recovery.

```text
schedule fused_elementwise [%3, %4, %5] elements=1024
```

This is an explicit lowering schedule over the existing expressions. Fusion does not replace the original arithmetic opcodes with an opaque blob. The printer exposes both the expression graph and the chosen loop region, which makes transformation tests and debugging straightforward.

DCE invalidates the schedule and reports that invalidation as a change, so it must run before fusion. The default pipeline ends with fusion; no final DCE is needed because scheduling introduces no new values. Callers modifying an already scheduled graph must rebuild its schedule. Stable IDs avoid dangling references during rewrites, but are not a general-purpose mutable IR framework.

Inspect with `tensorforge dump-ir FILE --opt --trace-passes`: the final IR goes to stdout and pass snapshots go to stderr. The interpreter ignores the schedule and evaluates live operations individually, so schedule annotations cannot silently turn the oracle into a fused execution path.

Tensor IR then lowers to the separate [Loop IR](loop-ir.md), which makes execution order and iteration structure explicit before LLVM construction.

Each public pass validates its incoming IR. Fusion invoked without DCE refuses a region whose final computation is not the returned value; the normal pipeline still runs DCE first.

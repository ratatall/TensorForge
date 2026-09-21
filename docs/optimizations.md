# Optimization legality

`PassManager` in `src/ir.cpp` runs three named functions in a fixed order and verifies IR after each. `--trace-passes` prints before/after IR and whether a pass reports a change. Exceptions include the failing pass's name. Optimization is opt-in for ordinary CLI execution; the benchmark constructs the four combinations of TensorForge passes off/on and LLVM none/O2.

## Constant folding

```text
before: %2 = multiply %0 %1 : f32  // constants 2 and 4
 after: %2 = constant 8 : f32
```

Legal when a live scalar add, multiply, or ReLU has only constant operands. Definition-order traversal makes a single pass enough for chains of constants. Evaluation uses the same binary32 operation and ReLU policy as the interpreter, without reassociation. Later DCE removes old constants that no longer have users.

Expected effect: remove runtime scalar computations and simplify the graph. LLVM's builder may independently fold literal arithmetic even on the unoptimized path, so this pass is primarily demonstrated through custom IR, not advertised as a measured speedup on its own. It does not fold input-dependent values or tensor constants (which the language does not have).

Tests in the `passes` group inspect a folded `8`, transitive ReLU/arithmetic folding, idempotence, and preservation of a live input-dependent multiply. Differential tests compare folded/fused compilation with the original oracle.

## Dead-code elimination

```text
let unused = A * 5;
return relu(A);
```

Only the returned expression graph is observable: all operations are pure and return is the sole sink. Seed liveness at the result and walk definitions in reverse, marking each live operation's operands. Every operand points backward, so one reverse traversal reaches all ancestors. Unmarked operations are hidden from printing and execution. The ABI input table stays intact so pointers keep their original meanings.

Expected effect: remove unused loops, buffer requirements, loads, and scalar computations. It cannot eliminate a reachable operation merely because its value looks numerically redundant. Removing dead floating-point operations also removes any floating-point exception flags they might raise; observing those flags is outside the language contract.

Tests check dead tensor multiplication disappears, live multiplication remains, folded operands become dead, and unused input removal does not renumber input arguments. The language has no side-effecting operations; adding one would require revisiting liveness roots and legality.

## Elementwise fusion

```text
before: multiply loop → temporary → add loop → temporary → relu loop → output
 after: one loop evaluates multiply, add, and relu for each output lane
```

Legal when all live tensor computations are supported pure elementwise operations with the same shape as the returned tensor. A scalar operand can be reused for every lane. Each output lane depends only on that lane of tensor inputs, so interleaving different operators per lane preserves the dependency graph. The arithmetic evaluation order *within* a lane is unchanged.

Fusion is a schedule, `fusedRegion`, over the existing SSA-like values. The code generator memoizes each scheduled value per lane. This handles linear chains, expression trees, and shared DAGs without recursively duplicating a shared producer. Scalar computations stay outside the loop. Output and scratch are host-owned and disjoint from inputs.

Expected effect: reduce full-array passes and intermediate memory traffic. A three-op ReLU chain uses three loops and two scratch tensors before fusion, one loop and zero scratch tensors after. A single operation also receives a one-op schedule but has no intermediate storage to eliminate; its timing should be nearly unchanged.

The fusion pass does not cross reductions or support reshapes, matrix multiplication, other cross-lane dependencies, side effects, or differently shaped live regions. Sum reductions lower through separate Loop IR nests; fusing a producer into one would need new legality and scheduling rules. Large DAGs can increase register pressure; removing scratch does not guarantee a speedup on every target.

Tests inspect the schedule and LLVM loop/scratch structure, then compare original and optimized execution across fixed and generated expression trees, several shapes/seeds, shared values, broadcasting, and ReLU edge cases. The primary custom-fusion comparison holds LLVM middle-end optimization at none. The benchmark separately reports both project settings with LLVM O2. See [LLVM optimization](llvm-optimization.md) for verification and generated-code evidence.

## Deliberately omitted algebraic rewrites

No `x + 0 → x` rewrite is implemented: it can change signed-zero behavior. Likewise `x * 0 → 0` fails for NaNs/infinities. Avoiding these tempting optional rewrites keeps the floating-point contract small and testable.

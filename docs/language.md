# TensorForge language

A program declares inputs and immutable names, then returns exactly one scalar or tensor. Inputs and `let` share one namespace. Declarations are ordered; forward references, shadowing, and statements after return are rejected.

```ebnf
program = (input_decl | let_decl)* "return" expression ";" EOF ;
input_decl = "input" identifier ":" type ";" ;
let_decl = "let" identifier "=" expression ";" ;
type = "f32" | "tensor" "<" positive_integer ("," positive_integer)* ">" ;
expression = product ("+" product)* ;
product = primary ("*" primary)* ;
primary = ["-"] number | identifier | "relu" "(" expression ")"
        | "(" expression ")" ;
```

`*` binds more tightly than `+`; both associate left. Unary minus is accepted only on numeric literals, not general expressions. There is no subtraction. Literals may contain a decimal point and `e`/`E` exponent, for example `2`, `.5`, `2.`, `1e-3`, and `-0.25`. Lexer whitespace is space, tab, CR, or LF. `//` runs to the end of a line. Identifiers use ASCII letters/underscore followed by letters, underscores, or digits. Keywords cannot be used as identifiers.

Dimensions must be integer spellings, not `4.0` or `4e0`, and each must be in `[1, 16777216]`. Tensor rank is limited to 8 and the product of all dimensions is limited to 16,777,216 elements. `tensor<1>` differs from a scalar in the type system, even though both contain one float. Integers used as expressions are rounded to `f32` like other numeric literals. Out-of-range large literals are rejected. Very small literals follow the host's classic-locale float conversion, including underflow to zero.

## Types and broadcasting

| Left | Right | `+` / `*` result |
|---|---|---|
| `f32` | `f32` | `f32` |
| `tensor<S...>` | `f32` | `tensor<S...>` |
| `f32` | `tensor<S...>` | `tensor<S...>` |
| `tensor<A...>` | `tensor<B...>` | broadcast shape, when compatible |
| incompatible tensor shapes | incompatible tensor shapes | diagnostic |

`relu` preserves its argument's type. Binary operations align tensor shapes from the trailing dimension. Two aligned dimensions are compatible when they are equal or either is 1; missing leading dimensions act as 1. For example, `tensor<2,3> + tensor<3>` produces `tensor<2,3>`, and `tensor<2,1> * tensor<1,3>` produces `tensor<2,3>`. Broadcast operands are indexed directly; no expanded tensor is allocated.

## Floating-point contract

Host execution uses IEEE-754 binary32 with the normal round-to-nearest environment. TensorForge does not change the floating-point environment. Do not enable fast-math or alter rounding/flush-to-zero modes when embedding it. Arithmetic overflow can produce infinity and invalid arithmetic can produce NaN. ReLU is exactly `x > 0.0f ? x : 0.0f`: it maps NaN and both signed zeros to positive zero, and preserves positive infinity. This policy is explicit, not necessarily identical to a framework's ReLU policy.

Host code is compiled with `-ffp-contract=off`; LLVM instructions carry no fast-math/contract flags. Folding does not change operation order. Tolerance arithmetic is evaluated in double precision to avoid comparison overflow; tolerances must be finite and nonnegative. The chosen 1e-6 absolute tolerance handles values near zero and 1e-5 relative tolerance handles f32 rounding at larger magnitudes. Verification accepts equal values (including same-sign infinities), pairs of NaNs, or finite values satisfying `abs(actual - expected) <= 1e-6 + 1e-5 * abs(expected)`. NaN payloads and signed-zero equality are not distinguished by the generic comparator; dedicated ReLU tests check its zero sign.

## Diagnostics and limits

Syntax and semantic diagnostics contain filename, one-based line/column, source line, and caret. Tabs are preserved in caret indentation. Nonprinting/non-ASCII source bytes are rendered as `\xNN`, including embedded NUL, so C-string diagnostics cannot truncate before the caret. Reported columns still refer to original byte positions. The compiler stops on the first error; it does not attempt recovery. Internal invariant failures are separate C++ exceptions, surfaced by the CLI as errors. Invalid input never proceeds from semantic analysis to LLVM.

Programs are limited to 16,384 tokens, 2,048 expression nodes, and 128 levels of parenthesized/call nesting. These bounds keep a recursive educational frontend manageable; they are not a security guarantee for hostile source files. See `examples/negative/` and `examples/shape_error.tf` for rejected programs.

Runtime resource budgets preflight generated input storage, interpreter materialization storage, and JIT scratch separately at 1 GiB each. Oversized aggregate programs can pass shape checking yet exceed an execution budget; rejection occurs before those buffers are allocated. LLVM O2 preserves this floating-point contract: no fast-math, reassociation, or contraction flags are enabled.

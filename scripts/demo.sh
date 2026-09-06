#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cli="${1:-build/tensorforge}"
"$cli" check examples/relu_chain.tf
cat examples/relu_chain.tf
"$cli" dump-ast examples/relu_chain.tf
"$cli" dump-ir examples/relu_chain.tf
"$cli" dump-ir examples/relu_chain.tf --opt
"$cli" dump-ir examples/scalar_folding.tf --opt
"$cli" dump-ir examples/dead_code.tf --opt
"$cli" emit-llvm examples/relu_chain.tf --opt
"$cli" emit-llvm examples/relu_chain.tf --opt --llvm-opt=O2
"$cli" run examples/relu_chain.tf --opt --llvm-opt=O2 --verify
"$cli" benchmark examples/relu_chain.tf --iterations 21

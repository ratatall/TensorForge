#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build_type="${1:-Release}"
build_dir="${2:-build}"
if [[ -z "${LLVM_DIR:-}" ]]; then
  if command -v llvm-config >/dev/null; then
    LLVM_DIR="$(llvm-config --cmakedir)"
  elif command -v llvm-config-23 >/dev/null; then
    LLVM_DIR="$(llvm-config-23 --cmakedir)"
  elif command -v brew >/dev/null && [[ -x "$(brew --prefix llvm)/bin/llvm-config" ]]; then
    LLVM_DIR="$($(brew --prefix llvm)/bin/llvm-config --cmakedir)"
  else
    echo 'Set LLVM_DIR to the LLVM 23 lib/cmake/llvm directory.' >&2
    exit 1
  fi
fi
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" -DLLVM_DIR="$LLVM_DIR"
cmake --build "$build_dir" --parallel "${JOBS:-4}"

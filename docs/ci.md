# Continuous integration

[The GitHub Actions workflow](../.github/workflows/linux.yml) targets Ubuntu 24.04 and LLVM 23 with two Release configurations: ordinary host code and ASan/UBSan-instrumented host code.

Each job builds with Ninja, runs all CTest cases, exercises the interpreter and four JIT configurations, verifies emitted LLVM IR with `opt-23`, and runs a benchmark smoke test. Failure logs are uploaded as artifacts. Repository permissions are read-only and checkout credentials are not persisted.

## Dependencies

`scripts/ci-install-llvm.sh` installs versioned LLVM/Clang 23 packages from the [official LLVM package repository](https://apt.llvm.org/). It selects the LLVM 23 release suite when available and otherwise the development suite. A version guard rejects any other major. Package versions within major 23 are not pinned.

## Platform coverage

Local validation covers macOS arm64 with LLVM 23.1.0. Linux validation is determined by the workflow results; local macOS checks do not establish Linux support. Only LLVM major 23 is currently accepted by CMake.

See [validation](validation.md) for test coverage and sanitizer scope.

# Testing and validation

Local validation used macOS 15.5 arm64, Apple Clang 17, LLVM 23.1.0, and CMake 4.4.3. All **19 CTest cases** passed in Debug, Release, and ASan/UBSan Release builds.

## Run the tests

```bash
scripts/build.sh Debug build-debug
scripts/test.sh build-debug
scripts/build.sh Release build-release
scripts/test.sh build-release
```

For sanitizer coverage, configure CMake with `-DTENSORFORGE_SANITIZERS=ON` and run CTest in the resulting build directory.

## Coverage

The suite covers lexical analysis, parsing, semantic checks, IR verification, optimization passes, code generation, differential execution, resource limits, runtime input validation, examples, negative CLI fixtures, CLI options, and benchmark output.

Differential tests perform **984 comparisons**: 60 fixed/generated graphs across four seeds and four JIT configurations, plus six shared DAGs across four configurations. Additional cases cover scalar broadcasting, dead code, constant folding, NaN/infinity/signed zero, malformed IR, overflow rejection, and mismatch diagnostics. LLVM functions and modules are verified before execution and after optional O2 optimization.

Release benchmark smoke tests validate finite timing data, configuration fields, loop and scratch counts, checksums, and verification status. The Debug case checks the Release-only benchmark requirement. Test assertions remain active in Release builds.

## Regression sensitivity and reproducibility

```bash
python3 scripts/mutation-check.py
python3 scripts/mutation-check.py --clean-only
```

The mutation check works in a temporary source copy. Disabling duplicate-name rejection is detected by the semantic tests; replacing LLVM addition with subtraction is detected by differential tests and CLI verification. Sources are restored between mutations. The clean-only mode builds and runs the suite and demo without existing build trees or generated results.

## Sanitizer scope

ASan/UBSan cover project host code, including repeated JIT construction and execution. Prebuilt LLVM and generated machine instructions are not instrumented. Apple's runtime does not support LeakSanitizer on the validated platform; Ubuntu CI requests leak detection. See [CI](ci.md) for the Linux configuration.

## Performance evidence

The checked-in benchmark CSV files contain two runs of nine workloads across the interpreter and four JIT configurations. Outputs are checked before and after timing. See [benchmarking](benchmarking.md) for methodology, machine details, and variability, and [LLVM optimization](llvm-optimization.md) for generated-code inspection.

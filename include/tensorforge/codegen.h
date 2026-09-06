#pragma once
#include "tensorforge/ir.h"
#include <memory>

namespace tensorforge {
enum class LLVMOptimization { None, O2 };
const char *optimizationName(LLVMOptimization level);
struct LoweringStats {
    std::size_t loops = 0, scratchElements = 0;
    std::size_t llvmLoops = 0, vectorInstructions = 0;
};
class Executable {
  public:
    explicit Executable(const Module &module, LLVMOptimization level = LLVMOptimization::None);
    ~Executable();
    Executable(Executable &&) noexcept;
    Executable &operator=(Executable &&) noexcept;
    Executable(const Executable &) = delete;
    Executable &operator=(const Executable &) = delete;
    Tensor run(const Inputs &inputs);
    // Prepared calls exclude validation and allocation. Inputs use declaration
    // order; output and scratch must be disjoint from inputs and each other.
    // Size scratch using stats().scratchElements; caller owns all buffers.
    void invoke(const float *const *inputs, float *output, float *scratch) const;
    const std::string &llvmIR() const;
    LoweringStats stats() const;
    const std::string &targetCPU() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::string llvmVersion();
} // namespace tensorforge

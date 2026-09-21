#pragma once

#include "tensorforge/ir.h"

namespace tensorforge {
enum class LoopKind { Scalar, Elementwise, Reduction, Copy, FusedElementwise };

struct LoopNest {
    LoopKind kind;
    std::vector<ValueId> operations;
    Type iterationType;
    std::size_t reductionAxis = 0;
    bool operator==(const LoopNest &) const = default;
};

struct LoopModule {
    std::vector<LoopNest> nests;
    bool operator==(const LoopModule &) const = default;
};

LoopModule lowerToLoopIR(const Module &module);
void validateLoopIR(const Module &module, const LoopModule &loops);
std::string printLoopIR(const Module &module, const LoopModule &loops);
} // namespace tensorforge

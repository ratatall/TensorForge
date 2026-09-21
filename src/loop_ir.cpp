#include "tensorforge/loop_ir.h"

#include <algorithm>
#include <sstream>

namespace tensorforge {
namespace {
const char *kindName(LoopKind kind) {
    switch (kind) {
    case LoopKind::Scalar:
        return "scalar";
    case LoopKind::Elementwise:
        return "elementwise";
    case LoopKind::Reduction:
        return "reduction";
    case LoopKind::Copy:
        return "copy";
    case LoopKind::FusedElementwise:
        return "fused_elementwise";
    }
    throw std::logic_error("unknown loop kind");
}

LoopModule build(const Module &module) {
    LoopModule result;
    const auto fused = [&](ValueId id) {
        return std::find(module.fusedRegion.begin(), module.fusedRegion.end(), id) !=
               module.fusedRegion.end();
    };
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (!op.alive || op.opcode == Opcode::Input || op.opcode == Opcode::Constant)
            continue;
        if (fused(id)) {
            if (id == module.result)
                result.nests.push_back(
                    {LoopKind::FusedElementwise, module.fusedRegion, op.type, 0});
            continue;
        }
        if (op.opcode == Opcode::ReduceSum)
            result.nests.push_back({LoopKind::Reduction, {id}, op.type, op.reductionAxis});
        else if (op.type.scalar())
            result.nests.push_back({LoopKind::Scalar, {id}, op.type, 0});
        else
            result.nests.push_back({LoopKind::Elementwise, {id}, op.type, 0});
    }
    const auto &returned = module.operations[module.result];
    if (returned.opcode == Opcode::Input && !returned.type.scalar())
        result.nests.push_back({LoopKind::Copy, {module.result}, returned.type, 0});
    return result;
}
} // namespace

LoopModule lowerToLoopIR(const Module &module) {
    validateIR(module);
    auto result = build(module);
    validateLoopIR(module, result);
    return result;
}

void validateLoopIR(const Module &module, const LoopModule &loops) {
    validateIR(module);
    const auto expected = build(module);
    if (loops != expected)
        throw std::logic_error("invalid Loop IR: schedule does not match Tensor IR");
}

std::string printLoopIR(const Module &module, const LoopModule &loops) {
    validateLoopIR(module, loops);
    std::ostringstream out;
    out << "loop_module {\n";
    for (std::size_t index = 0; index < loops.nests.size(); ++index) {
        const auto &nest = loops.nests[index];
        out << "  loop" << index << ' ' << kindName(nest.kind) << " [";
        for (std::size_t i = 0; i < nest.operations.size(); ++i)
            out << (i ? ", " : "") << '%' << nest.operations[i];
        out << "] over " << nest.iterationType.str();
        if (nest.kind == LoopKind::Reduction) {
            const auto &input =
                module.operations[module.operations[nest.operations[0]].operands[0]];
            out << " axis=" << nest.reductionAxis
                << " extent=" << input.type.shape[nest.reductionAxis];
        }
        out << '\n';
    }
    out << "}\n";
    return out.str();
}
} // namespace tensorforge

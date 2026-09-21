#include "tensorforge/ir.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace tensorforge {
bool broadcastTypes(const Type &left, const Type &right, Type &result) {
    const auto rank = std::max(left.shape.size(), right.shape.size());
    result.shape.assign(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto leftDimension =
            offset < left.shape.size() ? left.shape[left.shape.size() - 1 - offset] : 1;
        const auto rightDimension =
            offset < right.shape.size() ? right.shape[right.shape.size() - 1 - offset] : 1;
        if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1)
            return false;
        result.shape[rank - 1 - offset] = std::max(leftDimension, rightDimension);
    }
    return true;
}
std::size_t broadcastIndex(std::size_t outputIndex, const Type &operand, const Type &result) {
    if (operand.scalar())
        return 0;
    if (operand.shape.size() > result.shape.size())
        throw std::logic_error("broadcast operand rank exceeds result rank");
    std::size_t operandIndex = 0, operandStride = 1;
    for (std::size_t offset = 0; offset < result.shape.size(); ++offset) {
        const auto resultAxis = result.shape.size() - 1 - offset;
        const auto coordinate = outputIndex % result.shape[resultAxis];
        outputIndex /= result.shape[resultAxis];
        if (offset < operand.shape.size()) {
            const auto operandDimension = operand.shape[operand.shape.size() - 1 - offset];
            if (operandDimension != 1)
                operandIndex += coordinate * operandStride;
            operandStride *= operandDimension;
        }
    }
    return operandIndex;
}
bool reductionType(const Type &input, std::size_t axis, Type &result) {
    if (input.scalar() || axis >= input.shape.size())
        return false;
    result = input;
    result.shape.erase(result.shape.begin() + static_cast<std::ptrdiff_t>(axis));
    return true;
}
std::size_t reductionInputIndex(std::size_t outputIndex, std::size_t reductionIndex,
                                const Type &input, std::size_t axis) {
    Type output;
    if (!reductionType(input, axis, output))
        throw std::logic_error("invalid reduction shape or axis");
    std::size_t inputIndex = 0;
    for (std::size_t outputAxis = output.shape.size(); outputAxis-- > 0;) {
        const auto coordinate = outputIndex % output.shape[outputAxis];
        outputIndex /= output.shape[outputAxis];
        const auto inputAxis = outputAxis < axis ? outputAxis : outputAxis + 1;
        std::size_t stride = 1;
        for (std::size_t i = inputAxis + 1; i < input.shape.size(); ++i)
            stride *= input.shape[i];
        inputIndex += coordinate * stride;
    }
    std::size_t reductionStride = 1;
    for (std::size_t i = axis + 1; i < input.shape.size(); ++i)
        reductionStride *= input.shape[i];
    return inputIndex + reductionIndex * reductionStride;
}
namespace {
class Analyzer {
    const Source &source_;
    Module module_;
    std::unordered_map<std::string, ValueId> symbols_;
    ValueId append(Operation operation) {
        auto id = module_.operations.size();
        module_.operations.push_back(std::move(operation));
        return id;
    }
    ValueId expression(const Expr &expr) {
        return std::visit(
            [&](const auto &node) -> ValueId {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, Identifier>) {
                    auto found = symbols_.find(node.name);
                    if (found == symbols_.end())
                        source_.fail(expr.location, "undeclared identifier '" + node.name + "'");
                    return found->second;
                } else if constexpr (std::is_same_v<T, Number>) {
                    Operation op{Opcode::Constant, {}, {}, expr.location};
                    op.constant = node.value;
                    return append(std::move(op));
                } else if constexpr (std::is_same_v<T, Binary>) {
                    const auto left = expression(*node.left), right = expression(*node.right);
                    const auto a = module_.operations[left].type,
                               b = module_.operations[right].type;
                    Type result;
                    if (!broadcastTypes(a, b, result))
                        source_.fail(expr.location,
                                     "shape mismatch: " + a.str() + " versus " + b.str());
                    return append({node.op == '+' ? Opcode::Add : Opcode::Multiply,
                                   result,
                                   {left, right},
                                   expr.location});
                } else if constexpr (std::is_same_v<T, Relu>) {
                    const auto arg = expression(*node.argument);
                    return append(
                        {Opcode::Relu, module_.operations[arg].type, {arg}, expr.location});
                } else {
                    const auto arg = expression(*node.argument);
                    Type result;
                    if (!reductionType(module_.operations[arg].type, node.axis, result))
                        source_.fail(expr.location, "sum axis " + std::to_string(node.axis) +
                                                        " is invalid for " +
                                                        module_.operations[arg].type.str());
                    Operation op{Opcode::ReduceSum, result, {arg}, expr.location};
                    op.reductionAxis = node.axis;
                    return append(std::move(op));
                }
            },
            expr.node);
    }

  public:
    explicit Analyzer(const Source &source) : source_(source) {}
    Module run(const Program &program) {
        for (const auto &decl : program.declarations)
            std::visit(
                [&](const auto &node) {
                    if (symbols_.contains(node.name))
                        source_.fail(node.location, "duplicate declaration '" + node.name + "'");
                    using T = std::decay_t<decltype(node)>;
                    ValueId id;
                    if constexpr (std::is_same_v<T, InputDecl>) {
                        Operation op{Opcode::Input, node.type, {}, node.location};
                        op.name = node.name;
                        op.inputIndex = module_.inputs.size();
                        module_.inputs.push_back({node.name, node.type});
                        id = append(std::move(op));
                    } else
                        id = expression(*node.expression);
                    symbols_.emplace(node.name, id);
                },
                decl);
        module_.result = expression(*program.result.expression);
        validateIR(module_);
        return std::move(module_);
    }
};
const char *opcodeName(Opcode opcode) {
    switch (opcode) {
    case Opcode::Input:
        return "input";
    case Opcode::Constant:
        return "constant";
    case Opcode::Add:
        return "add";
    case Opcode::Multiply:
        return "multiply";
    case Opcode::Relu:
        return "relu";
    case Opcode::ReduceSum:
        return "reduce_sum";
    }
    throw std::logic_error("unknown IR opcode");
}
} // namespace
Module analyzeAndLower(const Source &source, const Program &program) {
    return Analyzer(source).run(program);
}
void validateIR(const Module &module) {
    auto invariant = [](bool ok, const char *message) {
        if (!ok)
            throw std::logic_error(std::string("invalid IR: ") + message);
    };
    invariant(module.result < module.operations.size(), "return value out of range");
    invariant(module.operations[module.result].alive, "return value is dead");
    auto validType = [](const Type &type) {
        if (type.shape.size() > MaxTensorRank)
            return false;
        std::size_t elements = 1;
        for (auto dimension : type.shape) {
            if (dimension == 0 || dimension > MaxTensorExtent ||
                elements > MaxTensorExtent / dimension)
                return false;
            elements *= dimension;
        }
        return true;
    };
    for (const auto &input : module.inputs)
        invariant(validType(input.type), "input shape exceeds limit");
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (!op.alive)
            continue;
        invariant(validType(op.type), "result shape exceeds limit");
        switch (op.opcode) {
        case Opcode::Input:
        case Opcode::Constant:
        case Opcode::Add:
        case Opcode::Multiply:
        case Opcode::Relu:
        case Opcode::ReduceSum:
            break;
        default:
            invariant(false, "unknown opcode");
        }
        for (auto arg : op.operands)
            invariant(arg < id && module.operations[arg].alive,
                      "operand must be a live preceding definition");
        const auto arity =
            (op.opcode == Opcode::Relu || op.opcode == Opcode::ReduceSum)
                ? 1U
                : (op.opcode == Opcode::Add || op.opcode == Opcode::Multiply ? 2U : 0U);
        invariant(op.operands.size() == arity, "wrong operand count");
        if (op.opcode == Opcode::Input) {
            invariant(op.inputIndex < module.inputs.size(), "invalid input index");
            invariant(op.type == module.inputs[op.inputIndex].type, "input type mismatch");
        } else if (op.opcode == Opcode::Constant)
            invariant(op.type.scalar(), "constant must be scalar");
        else if (op.opcode == Opcode::ReduceSum) {
            Type result;
            invariant(
                reductionType(module.operations[op.operands[0]].type, op.reductionAxis, result),
                "invalid reduction shape or axis");
            invariant(result == op.type, "incorrect reduction result type");
        } else {
            Type result;
            invariant(op.operands.empty() ||
                          broadcastTypes(module.operations[op.operands[0]].type,
                                         op.opcode == Opcode::Relu
                                             ? module.operations[op.operands[0]].type
                                             : module.operations[op.operands[1]].type,
                                         result),
                      "operand shape mismatch");
            invariant(result == op.type, "incorrect result type");
        }
    }
    if (!module.fusedRegion.empty()) {
        for (const auto &op : module.operations)
            invariant(!op.alive || op.opcode != Opcode::ReduceSum,
                      "fusion does not support reductions");
        std::vector<ValueId> expected;
        for (ValueId id = 0; id < module.operations.size(); ++id) {
            const auto &op = module.operations[id];
            if (op.alive && !op.type.scalar() && op.opcode != Opcode::Input) {
                invariant(op.type == module.operations[module.result].type,
                          "fusion shape mismatch");
                expected.push_back(id);
            }
        }
        invariant(expected == module.fusedRegion,
                  "fusion schedule must contain all live tensor computations in definition order");
        invariant(module.fusedRegion.back() == module.result, "fusion must end at return");
    }
}
std::string printIR(const Module &module) {
    validateIR(module);
    std::ostringstream out;
    out << "module {\n" << std::setprecision(9);
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (!op.alive)
            continue;
        out << "  %" << id << " = " << opcodeName(op.opcode);
        if (op.opcode == Opcode::Input)
            out << " \"" << op.name << "\" [argument " << op.inputIndex << ']';
        if (op.opcode == Opcode::Constant)
            out << ' ' << op.constant;
        if (op.opcode == Opcode::ReduceSum)
            out << " axis=" << op.reductionAxis;
        for (auto arg : op.operands)
            out << " %" << arg;
        out << " : " << op.type.str() << '\n';
    }
    if (!module.fusedRegion.empty()) {
        out << "  schedule fused_elementwise [";
        for (std::size_t i = 0; i < module.fusedRegion.size(); ++i)
            out << (i ? ", " : "") << '%' << module.fusedRegion[i];
        out << "] elements=" << module.operations[module.result].type.elements() << '\n';
    }
    out << "  return %" << module.result << "\n}\n";
    return out.str();
}
float applyRelu(float value) {
    return value > 0.0f ? value : 0.0f;
}
bool constantFold(Module &module) {
    validateIR(module);
    bool changed = false;
    for (auto &op : module.operations) {
        if (!op.alive || !op.type.scalar() || op.operands.empty())
            continue;
        bool constants = true;
        for (auto arg : op.operands)
            constants &= module.operations[arg].opcode == Opcode::Constant;
        if (!constants)
            continue;
        const float a = module.operations[op.operands[0]].constant;
        if (op.opcode == Opcode::Relu)
            op.constant = applyRelu(a);
        else {
            const float b = module.operations[op.operands[1]].constant;
            op.constant = op.opcode == Opcode::Add ? a + b : a * b;
        }
        op.opcode = Opcode::Constant;
        op.operands.clear();
        changed = true;
    }
    return changed;
}
bool eliminateDeadCode(Module &module) {
    validateIR(module);
    std::vector<bool> live(module.operations.size(), false);
    live.at(module.result) = true;
    // SSA definitions precede uses, so one reverse walk is sufficient.
    for (std::size_t id = module.operations.size(); id-- > 0;)
        if (live[id])
            for (auto arg : module.operations[id].operands)
                live[arg] = true;
    bool changed = !module.fusedRegion.empty();
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        changed |= module.operations[id].alive != live[id];
        module.operations[id].alive = live[id];
    }
    module.fusedRegion.clear();
    return changed;
}
bool fuseElementwise(Module &module) {
    validateIR(module);
    std::vector<ValueId> region;
    // Caller runs DCE first. Fusion currently requires one common tensor shape.
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (op.alive && op.opcode == Opcode::ReduceSum)
            return false;
        if (op.alive && !op.type.scalar() && op.opcode != Opcode::Input) {
            if (op.type != module.operations[module.result].type)
                return false;
            region.push_back(id);
        }
    }
    // A caller may invoke this pass without DCE first. Do not install a
    // schedule whose final computation is not the returned value.
    if (!region.empty() && region.back() != module.result)
        return false;
    const bool changed = region != module.fusedRegion;
    module.fusedRegion = std::move(region);
    return changed;
}
PassManager::PassManager()
    : passes_{{"constant-folding", constantFold},
              {"dead-code-elimination", eliminateDeadCode},
              {"elementwise-fusion", fuseElementwise}} {}
void PassManager::run(Module &module, std::ostream *trace) const {
    validateIR(module);
    for (const auto &pass : passes_) {
        if (trace)
            *trace << "before " << pass.name << '\n' << printIR(module);
        try {
            const bool changed = pass.run(module);
            validateIR(module);
            if (trace)
                *trace << "after " << pass.name << " (" << (changed ? "changed" : "unchanged")
                       << ")\n"
                       << printIR(module);
        } catch (const std::exception &error) {
            throw std::runtime_error("pass '" + pass.name + "' failed: " + error.what());
        }
    }
}
} // namespace tensorforge

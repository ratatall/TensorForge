#include "tensorforge/ir.h"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace tensorforge {
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
                    if (!a.scalar() && !b.scalar() && a != b)
                        source_.fail(expr.location,
                                     "shape mismatch: " + a.str() + " versus " + b.str());
                    return append({node.op == '+' ? Opcode::Add : Opcode::Multiply,
                                   a.scalar() ? b : a,
                                   {left, right},
                                   expr.location});
                } else {
                    const auto arg = expression(*node.argument);
                    return append(
                        {Opcode::Relu, module_.operations[arg].type, {arg}, expr.location});
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
    for (const auto &input : module.inputs)
        invariant(input.type.extent <= MaxTensorExtent, "input extent exceeds limit");
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (!op.alive)
            continue;
        invariant(op.type.extent <= MaxTensorExtent, "result extent exceeds limit");
        switch (op.opcode) {
        case Opcode::Input:
        case Opcode::Constant:
        case Opcode::Add:
        case Opcode::Multiply:
        case Opcode::Relu:
            break;
        default:
            invariant(false, "unknown opcode");
        }
        for (auto arg : op.operands)
            invariant(arg < id && module.operations[arg].alive,
                      "operand must be a live preceding definition");
        const auto arity =
            op.opcode == Opcode::Relu
                ? 1U
                : (op.opcode == Opcode::Add || op.opcode == Opcode::Multiply ? 2U : 0U);
        invariant(op.operands.size() == arity, "wrong operand count");
        if (op.opcode == Opcode::Input) {
            invariant(op.inputIndex < module.inputs.size(), "invalid input index");
            invariant(op.type == module.inputs[op.inputIndex].type, "input type mismatch");
        } else if (op.opcode == Opcode::Constant)
            invariant(op.type.scalar(), "constant must be scalar");
        else {
            Type result;
            for (auto arg : op.operands) {
                const auto type = module.operations[arg].type;
                invariant(result.scalar() || type.scalar() || result == type,
                          "operand shape mismatch");
                if (!type.scalar())
                    result = type;
            }
            invariant(result == op.type, "incorrect result type");
        }
    }
    if (!module.fusedRegion.empty()) {
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
        for (auto arg : op.operands)
            out << " %" << arg;
        out << " : " << op.type.str() << '\n';
    }
    if (!module.fusedRegion.empty()) {
        out << "  schedule fused_elementwise [";
        for (std::size_t i = 0; i < module.fusedRegion.size(); ++i)
            out << (i ? ", " : "") << '%' << module.fusedRegion[i];
        out << "] extent=" << module.operations[module.result].type.extent << '\n';
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
    // Caller runs DCE first. A live rank-1 elementwise graph has one extent.
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
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

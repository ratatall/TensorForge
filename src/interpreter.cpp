#include "tensorforge/ir.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace tensorforge {
std::size_t checkedFloatBytes(std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("f32 buffer byte size overflow");
    return elements * sizeof(float);
}
std::size_t addBufferElements(std::size_t total, std::size_t elements) {
    checkedFloatBytes(elements);
    if (total > MaxBufferElements || elements > MaxBufferElements - total)
        throw std::length_error("buffer requirement exceeds 1 GiB limit");
    return total + elements;
}
void validateInputs(const Module &module, const Inputs &inputs) {
    if (inputs.size() != module.inputs.size())
        throw std::invalid_argument("input count mismatch");
    for (std::size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].size() != module.inputs[i].type.elements())
            throw std::invalid_argument("input shape mismatch: " + module.inputs[i].name);
}
Inputs generateInputs(const Module &module, std::uint32_t seed) {
    validateIR(module);
    std::size_t total = 0;
    for (const auto &input : module.inputs)
        total = addBufferElements(total, input.type.elements());
    std::mt19937 generator(seed);
    Inputs inputs;
    for (const auto &input : module.inputs) {
        Tensor values(input.type.elements());
        // Fixed integer-to-float mapping, unlike implementation-specific uniform_real_distribution.
        for (auto &value : values)
            value = static_cast<float>(static_cast<int>(generator() % 4097) - 2048) / 512.0f;
        inputs.push_back(std::move(values));
    }
    return inputs;
}
Tensor interpret(const Module &module, const Inputs &inputs) {
    validateIR(module);
    validateInputs(module, inputs);
    // Preflight the full materialization budget before allocating any tensors.
    std::size_t total = 0;
    for (const auto &op : module.operations)
        if (op.alive)
            total = addBufferElements(total, op.type.elements());
    std::vector<Tensor> values(module.operations.size());
    for (ValueId id = 0; id < module.operations.size(); ++id) {
        const auto &op = module.operations[id];
        if (!op.alive)
            continue;
        auto &result = values[id];
        if (op.opcode == Opcode::Input) {
            result = inputs.at(op.inputIndex);
            continue;
        }
        if (op.opcode == Opcode::Constant) {
            result = {op.constant};
            continue;
        }
        result.resize(op.type.elements());
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto get = [&](std::size_t operand) {
                const auto arg = op.operands.at(operand);
                return values.at(arg).at(module.operations[arg].type.scalar() ? 0 : i);
            };
            const float a = get(0);
            switch (op.opcode) {
            case Opcode::Add:
                result[i] = a + get(1);
                break;
            case Opcode::Multiply:
                result[i] = a * get(1);
                break;
            case Opcode::Relu:
                result[i] = applyRelu(a);
                break;
            default:
                throw std::logic_error("interpreter: invalid computation");
            }
        }
    }
    return values.at(module.result);
}
void compare(const Tensor &actual, const Tensor &expected, float atol, float rtol) {
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
        throw std::invalid_argument("verification tolerances must be finite and nonnegative");
    if (actual.size() != expected.size())
        throw std::runtime_error("verification failed: output shape mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i], e = expected[i];
        const bool equal =
            a == e || (std::isnan(a) && std::isnan(e)) ||
            (std::isfinite(a) && std::isfinite(e) &&
             std::abs(static_cast<double>(a) - e) <=
                 static_cast<double>(atol) + static_cast<double>(rtol) * std::abs(e));
        if (!equal) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<float>::max_digits10)
                    << "verification failed at element " << i << ": actual=" << a
                    << ", expected=" << e;
            throw std::runtime_error(message.str());
        }
    }
}
} // namespace tensorforge

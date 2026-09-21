#pragma once
#include "tensorforge/frontend.h"
#include <cstdint>
#include <functional>
#include <ostream>

namespace tensorforge {
using ValueId = std::size_t;
enum class Opcode { Input, Constant, Add, Multiply, Relu, ReduceSum };
struct Operation {
    Opcode opcode;
    Type type;
    std::vector<ValueId> operands;
    Location location;
    float constant = 0;
    std::string name;
    std::size_t inputIndex = 0;
    std::size_t reductionAxis = 0;
    bool alive = true;
};
struct InputInfo {
    std::string name;
    Type type;
};
struct Module {
    std::vector<Operation> operations;
    std::vector<InputInfo> inputs;
    ValueId result = 0;
    // A scheduled region: every member is evaluated once per output element.
    std::vector<ValueId> fusedRegion;
};
bool broadcastTypes(const Type &left, const Type &right, Type &result);
std::size_t broadcastIndex(std::size_t outputIndex, const Type &operand, const Type &result);
bool reductionType(const Type &input, std::size_t axis, Type &result);
std::size_t reductionInputIndex(std::size_t outputIndex, std::size_t reductionIndex,
                                const Type &input, std::size_t axis);
Module analyzeAndLower(const Source &source, const Program &program);
void validateIR(const Module &module);
std::string printIR(const Module &module);
bool constantFold(Module &module);
bool eliminateDeadCode(Module &module);
bool fuseElementwise(Module &module);
struct Pass {
    std::string name;
    std::function<bool(Module &)> run;
};
class PassManager {
  public:
    PassManager();
    void run(Module &module, std::ostream *trace = nullptr) const;

  private:
    std::vector<Pass> passes_;
};
using Tensor = std::vector<float>;
using Inputs = std::vector<Tensor>;
inline constexpr std::size_t MaxBufferElements = 268435456; // 1 GiB of f32.
std::size_t checkedFloatBytes(std::size_t elements);
std::size_t addBufferElements(std::size_t total, std::size_t elements);
Inputs generateInputs(const Module &module, std::uint32_t seed);
Tensor interpret(const Module &module, const Inputs &inputs);
void validateInputs(const Module &module, const Inputs &inputs);
void compare(const Tensor &actual, const Tensor &expected, float atol = 1e-6f, float rtol = 1e-5f);
float applyRelu(float value);
} // namespace tensorforge

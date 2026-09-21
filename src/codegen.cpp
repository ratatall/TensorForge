#include "tensorforge/codegen.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <mutex>

namespace tensorforge {
namespace {
using Kernel = void (*)(const float *const *, float *, float *);
void checkError(llvm::Error error) {
    if (error)
        throw std::runtime_error("LLVM: " + llvm::toString(std::move(error)));
}
template <class T> T unwrap(llvm::Expected<T> value) {
    if (!value)
        checkError(value.takeError());
    return std::move(*value);
}
class Generator {
    const Module &ir_;
    llvm::LLVMContext &context_;
    llvm::Module &module_;
    llvm::IRBuilder<> builder_;
    llvm::Function *function_ = nullptr;
    llvm::Value *output_ = nullptr, *scratch_ = nullptr;
    std::vector<llvm::Value *> pointers_, scalars_;
    LoweringStats stats_;
    llvm::Type *f32() { return builder_.getFloatTy(); }
    llvm::Value *offset(llvm::Value *pointer, llvm::Value *index) {
        return builder_.CreateGEP(f32(), pointer, index);
    }
    llvm::Value *operation(const Operation &op,
                           const std::function<llvm::Value *(ValueId, const Type &)> &get) {
        auto *a = get(op.operands.at(0), op.type);
        if (op.opcode == Opcode::Add)
            return builder_.CreateFAdd(a, get(op.operands.at(1), op.type), "add");
        if (op.opcode == Opcode::Multiply)
            return builder_.CreateFMul(a, get(op.operands.at(1), op.type), "multiply");
        if (op.opcode == Opcode::Relu) {
            auto *zero = llvm::ConstantFP::get(f32(), 0.0);
            // Ordered comparison maps NaNs and both signed zeros to positive zero.
            return builder_.CreateSelect(builder_.CreateFCmpOGT(a, zero, "positive"), a, zero,
                                         "relu");
        }
        throw std::logic_error("codegen: unsupported computation");
    }
    llvm::Value *load(ValueId id, llvm::Value *index, const Type &resultType) {
        if (ir_.operations[id].type.scalar())
            return scalars_.at(id);
        if (!pointers_.at(id))
            throw std::logic_error("codegen: missing tensor buffer");
        const auto &operandType = ir_.operations[id].type;
        llvm::Value *operandIndex = index;
        if (operandType != resultType) {
            operandIndex = builder_.getInt64(0);
            llvm::Value *remaining = index;
            std::size_t operandStride = 1;
            for (std::size_t axisOffset = 0; axisOffset < resultType.shape.size(); ++axisOffset) {
                const auto resultAxis = resultType.shape.size() - 1 - axisOffset;
                auto *dimension = builder_.getInt64(resultType.shape[resultAxis]);
                auto *coordinate = builder_.CreateURem(remaining, dimension, "coordinate");
                remaining = builder_.CreateUDiv(remaining, dimension, "remaining");
                if (axisOffset < operandType.shape.size()) {
                    const auto operandDimension =
                        operandType.shape[operandType.shape.size() - 1 - axisOffset];
                    if (operandDimension != 1) {
                        auto *term = builder_.CreateMul(
                            coordinate, builder_.getInt64(operandStride), "broadcast.offset");
                        operandIndex = builder_.CreateAdd(operandIndex, term, "broadcast.index");
                    }
                    operandStride *= operandDimension;
                }
            }
        }
        return builder_.CreateLoad(f32(), offset(pointers_[id], operandIndex),
                                   "v" + std::to_string(id));
    }
    void loop(std::size_t extent, const std::function<void(llvm::Value *)> &body) {
        ++stats_.loops;
        auto *before = builder_.GetInsertBlock();
        auto *block = llvm::BasicBlock::Create(context_, "loop", function_);
        auto *after = llvm::BasicBlock::Create(context_, "after", function_);
        builder_.CreateBr(block);
        builder_.SetInsertPoint(block);
        auto *index = builder_.CreatePHI(builder_.getInt64Ty(), 2, "i");
        index->addIncoming(builder_.getInt64(0), before);
        body(index);
        auto *next = builder_.CreateAdd(index, builder_.getInt64(1), "next");
        index->addIncoming(next, block);
        builder_.CreateCondBr(builder_.CreateICmpULT(next, builder_.getInt64(extent)), block,
                              after);
        builder_.SetInsertPoint(after);
    }

  public:
    Generator(const Module &ir, llvm::LLVMContext &context, llvm::Module &module)
        : ir_(ir), context_(context), module_(module), builder_(context),
          pointers_(ir.operations.size()), scalars_(ir.operations.size()) {}
    LoweringStats run() {
        auto *ptr = llvm::PointerType::getUnqual(context_);
        auto *signature = llvm::FunctionType::get(builder_.getVoidTy(), {ptr, ptr, ptr}, false);
        function_ = llvm::Function::Create(signature, llvm::Function::ExternalLinkage,
                                           "tensorforge_run", module_);
        auto *inputs = function_->getArg(0);
        inputs->setName("inputs");
        output_ = function_->getArg(1);
        output_->setName("output");
        scratch_ = function_->getArg(2);
        scratch_->setName("scratch");
        // The public ABI requires these buffers to be disjoint from all inputs.
        function_->addParamAttr(1, llvm::Attribute::NoAlias);
        function_->addParamAttr(2, llvm::Attribute::NoAlias);
        builder_.SetInsertPoint(llvm::BasicBlock::Create(context_, "entry", function_));
        const bool fused = !ir_.fusedRegion.empty();
        // Allocate on the host; the generated kernel never calls malloc or uses large stack arrays.
        for (ValueId id = 0; id < ir_.operations.size(); ++id) {
            const auto &op = ir_.operations[id];
            if (!op.alive)
                continue;
            if (op.opcode == Opcode::Input) {
                auto *slot = builder_.CreateGEP(ptr, inputs, builder_.getInt64(op.inputIndex));
                pointers_[id] = builder_.CreateLoad(ptr, slot, op.name);
            } else if (!op.type.scalar() && !fused) {
                if (id == ir_.result)
                    pointers_[id] = output_;
                else {
                    const auto nextScratch =
                        addBufferElements(stats_.scratchElements, op.type.elements());
                    pointers_[id] = offset(scratch_, builder_.getInt64(stats_.scratchElements));
                    stats_.scratchElements = nextScratch;
                }
            }
        }
        for (ValueId id = 0; id < ir_.operations.size(); ++id) {
            const auto &op = ir_.operations[id];
            if (!op.alive)
                continue;
            if (op.type.scalar()) {
                if (op.opcode == Opcode::Input)
                    scalars_[id] = builder_.CreateLoad(f32(), pointers_[id], op.name + ".scalar");
                else if (op.opcode == Opcode::Constant)
                    scalars_[id] = llvm::ConstantFP::get(f32(), op.constant);
                else
                    scalars_[id] =
                        operation(op, [&](ValueId arg, const Type &) { return scalars_.at(arg); });
            } else if (!fused && op.opcode != Opcode::Input) {
                loop(op.type.elements(), [&](llvm::Value *index) {
                    auto *value = operation(
                        op, [&](ValueId arg, const Type &type) { return load(arg, index, type); });
                    builder_.CreateStore(value, offset(pointers_[id], index));
                });
            }
        }
        const auto &result = ir_.operations[ir_.result];
        if (fused) {
            loop(result.type.elements(), [&](llvm::Value *index) {
                auto values = scalars_;
                // Cache DAG values in SSA registers, including shared subexpressions.
                auto get = [&](ValueId arg, const Type &type) {
                    if (!values[arg])
                        values[arg] = load(arg, index, type);
                    return values[arg];
                };
                for (auto id : ir_.fusedRegion)
                    values[id] = operation(ir_.operations[id], get);
                builder_.CreateStore(values.at(ir_.result), offset(output_, index));
            });
        } else if (result.type.scalar())
            builder_.CreateStore(scalars_.at(ir_.result), output_);
        else if (result.opcode == Opcode::Input) {
            loop(result.type.elements(), [&](llvm::Value *index) {
                builder_.CreateStore(load(ir_.result, index, result.type), offset(output_, index));
            });
        }
        builder_.CreateRetVoid();
        std::string message;
        llvm::raw_string_ostream errors(message);
        const bool invalidFunction = llvm::verifyFunction(*function_, &errors);
        const bool invalidModule = llvm::verifyModule(module_, &errors);
        if (invalidFunction || invalidModule)
            throw std::runtime_error("LLVM verification failed: " + message);
        return stats_;
    }
};
} // namespace
struct Executable::Impl {
    Module ir;
    std::unique_ptr<llvm::orc::LLJIT> jit;
    Kernel kernel = nullptr;
    std::string text;
    std::string cpu;
    LoweringStats stats;
};
Executable::Executable(const Module &module, LLVMOptimization level)
    : impl_(std::make_unique<Impl>()) {
    if (level != LLVMOptimization::None && level != LLVMOptimization::O2)
        throw std::invalid_argument("unsupported LLVM optimization level");
    validateIR(module);
    impl_->ir = module;
    // LLVM native registration is process-global; call_once is the only global state.
    static std::once_flag initialize;
    std::call_once(initialize, [] {
        if (llvm::InitializeNativeTarget() || llvm::InitializeNativeTargetAsmPrinter())
            throw std::runtime_error("LLVM native target initialization failed");
    });
    auto target = unwrap(llvm::orc::JITTargetMachineBuilder::detectHost());
    // Keep machine-code optimization fixed while varying the middle end.
    target.setCodeGenOptLevel(llvm::CodeGenOptLevel::Default);
    auto machine = unwrap(target.createTargetMachine());
    impl_->cpu = target.getCPU();
    impl_->jit =
        unwrap(llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(target)).create());
    // O2 can form memcpy/memset intrinsics that the backend lowers to libc calls.
    impl_->jit->getMainJITDylib().addGenerator(
        unwrap(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            impl_->jit->getDataLayout().getGlobalPrefix())));
    auto context = std::make_unique<llvm::LLVMContext>();
    auto generated = std::make_unique<llvm::Module>("tensorforge", *context);
    generated->setDataLayout(impl_->jit->getDataLayout());
    generated->setTargetTriple(impl_->jit->getTargetTriple());
    impl_->stats = Generator(module, *context, *generated).run();
    auto *function = generated->getFunction("tensorforge_run");
    function->addFnAttr("target-cpu", machine->getTargetCPU());
    function->addFnAttr("target-features", machine->getTargetFeatureString());
    // Generator verified the original function and module. Run the standard
    // new-PM pipeline, then verify again before printing or handing code to ORC.
    if (level == LLVMOptimization::O2) {
        llvm::LoopAnalysisManager loops;
        llvm::FunctionAnalysisManager functions;
        llvm::CGSCCAnalysisManager cgscc;
        llvm::ModuleAnalysisManager modules;
        llvm::PassBuilder passes(machine.get());
        passes.registerModuleAnalyses(modules);
        passes.registerCGSCCAnalyses(cgscc);
        passes.registerFunctionAnalyses(functions);
        passes.registerLoopAnalyses(loops);
        passes.crossRegisterProxies(loops, functions, cgscc, modules);
        auto pipeline = passes.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
        pipeline.run(*generated, modules);
    }
    std::string errors;
    llvm::raw_string_ostream errorStream(errors);
    const bool badFunction = llvm::verifyFunction(*function, &errorStream);
    const bool badModule = llvm::verifyModule(*generated, &errorStream);
    if (badFunction || badModule)
        throw std::runtime_error("LLVM verification after optimization failed: " + errors);
    llvm::DominatorTree dominators(*function);
    llvm::LoopInfo loops(dominators);
    impl_->stats.llvmLoops = loops.getLoopsInPreorder().size();
    for (const auto &block : *function)
        for (const auto &instruction : block)
            if (instruction.getType()->isVectorTy())
                ++impl_->stats.vectorInstructions;
    llvm::raw_string_ostream text(impl_->text);
    generated->print(text, nullptr);
    checkError(impl_->jit->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(generated), std::move(context))));
    // ExecutorAddr::toPtr is LLVM's supported in-process address conversion.
    impl_->kernel = unwrap(impl_->jit->lookup("tensorforge_run")).toPtr<Kernel>();
}
Executable::~Executable() = default;
Executable::Executable(Executable &&) noexcept = default;
Executable &Executable::operator=(Executable &&) noexcept = default;
#if defined(__clang__)
// Clang's UBSan "function" check reads type metadata immediately before an indirect
// callee. ORC-generated functions do not carry that host-compiler metadata and may
// begin at a page boundary, so the check itself can fault at address (kernel - 8).
// Keep every other ASan/UBSan check active across this narrow JIT ABI boundary.
__attribute__((no_sanitize("function")))
#endif
void Executable::invoke(const float *const *inputs, float *output, float *scratch) const {
    impl_->kernel(inputs, output, scratch);
}
Tensor Executable::run(const Inputs &inputs) {
    validateInputs(impl_->ir, inputs);
    std::vector<const float *> pointers;
    for (const auto &input : inputs)
        pointers.push_back(input.data());
    Tensor output(impl_->ir.operations[impl_->ir.result].type.elements()),
        scratch(impl_->stats.scratchElements);
    invoke(pointers.data(), output.data(), scratch.data());
    return output;
}
const std::string &Executable::llvmIR() const {
    return impl_->text;
}
LoweringStats Executable::stats() const {
    return impl_->stats;
}
const std::string &Executable::targetCPU() const {
    return impl_->cpu;
}
const char *optimizationName(LLVMOptimization level) {
    return level == LLVMOptimization::O2 ? "O2" : "none";
}
std::string llvmVersion() {
    return LLVM_VERSION_STRING;
}
} // namespace tensorforge

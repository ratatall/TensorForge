#include "tensorforge/benchmark.h"
#include <charconv>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
std::uint32_t integer(const std::string &text) {
    std::uint32_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument("expected unsigned 32-bit integer: " + text);
    return value;
}
void usage() {
    std::cout << "TensorForge 0.1.0\n"
                 "  tensorforge check FILE\n  tensorforge dump-ast FILE\n"
                 "  tensorforge dump-ir FILE [--opt] [--trace-passes]\n"
                 "  tensorforge emit-llvm FILE [--opt] [--llvm-opt=none|O2]\n"
                 "  tensorforge run FILE [--opt] [--seed N] [--verify] [--interpret] "
                 "[--llvm-opt=none|O2]\n"
                 "  tensorforge benchmark FILE [--iterations N] [--seed N] [--csv FILE]\n";
}
} // namespace
int main(int argc, char **argv) {
    using namespace tensorforge;
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage();
            return 0;
        }
        if (argc < 3) {
            usage();
            return 2;
        }
        const std::string command = argv[1], path = argv[2];
        if (command != "check" && command != "dump-ast" && command != "dump-ir" &&
            command != "emit-llvm" && command != "run" && command != "benchmark")
            throw std::invalid_argument("unknown command: " + command);
        bool opt = false, verify = false, interpreter = false, trace = false;
        LLVMOptimization llvmOpt = LLVMOptimization::None;
        std::uint32_t seed = 42, iterations = 101;
        std::string csv;
        for (int i = 3; i < argc; ++i) {
            const std::string flag = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc)
                    throw std::invalid_argument("missing value for " + flag);
                return argv[i];
            };
            if (flag.starts_with("--llvm-opt=") && (command == "run" || command == "emit-llvm")) {
                const auto level = flag.substr(11);
                if (level == "none")
                    llvmOpt = LLVMOptimization::None;
                else if (level == "O2")
                    llvmOpt = LLVMOptimization::O2;
                else
                    throw std::invalid_argument("--llvm-opt must be none or O2");
            } else if (flag == "--opt" &&
                       (command == "dump-ir" || command == "emit-llvm" || command == "run"))
                opt = true;
            else if (flag == "--verify" && command == "run")
                verify = true;
            else if (flag == "--interpret" && command == "run")
                interpreter = true;
            else if (flag == "--trace-passes" && command == "dump-ir")
                trace = true;
            else if (flag == "--seed" && (command == "run" || command == "benchmark"))
                seed = integer(value());
            else if (flag == "--iterations" && command == "benchmark")
                iterations = integer(value());
            else if (flag == "--csv" && command == "benchmark")
                csv = value();
            else
                throw std::invalid_argument("unknown or inapplicable option: " + flag);
        }
        if (trace && !opt)
            throw std::invalid_argument("--trace-passes requires --opt");
        if (interpreter && llvmOpt != LLVMOptimization::None)
            throw std::invalid_argument("--interpret requires --llvm-opt=none");
        if (interpreter && opt)
            throw std::invalid_argument("--interpret uses unoptimized reference IR; omit --opt");
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error("cannot open " + path);
        std::ostringstream contents;
        contents << file.rdbuf();
        Source source{path, contents.str()};
        auto ast = parse(source);
        if (command == "dump-ast") {
            std::cout << printAST(ast);
            return 0;
        }
        auto original = analyzeAndLower(source, ast);
        if (command == "check") {
            std::cout << path << ": OK\n";
            return 0;
        }
        if (command == "benchmark") {
            benchmark(original, iterations, seed, csv, path);
            return 0;
        }
        auto ir = original;
        if (opt)
            PassManager().run(ir, trace ? &std::cerr : nullptr);
        if (command == "dump-ir") {
            std::cout << printIR(ir);
            return 0;
        }
        if (command == "emit-llvm") {
            Executable executable(ir, llvmOpt);
            std::cout << executable.llvmIR();
            return 0;
        }
        const auto inputs = generateInputs(original, seed);
        Tensor result;
        if (interpreter)
            result = interpret(original, inputs);
        else {
            Executable executable(ir, llvmOpt);
            result = executable.run(inputs);
        }
        if (verify) {
            compare(result, interpret(original, inputs));
            // Also compare both JIT schedules, even when --interpret was selected.
            auto optimized = original;
            PassManager().run(optimized);
            for (auto level : {LLVMOptimization::None, LLVMOptimization::O2}) {
                compare(Executable(original, level).run(inputs), result);
                compare(Executable(optimized, level).run(inputs), result);
            }
        }
        std::cout << "result (" << result.size() << " element(s)): [";
        for (std::size_t i = 0; i < std::min<std::size_t>(result.size(), 8); ++i)
            std::cout << (i ? ", " : "") << result[i];
        std::cout << (result.size() > 8 ? ", ..." : "") << "]\n";
        if (verify)
            std::cout << "verified: interpreter == all 4 JIT configurations (TensorForge off/on, "
                         "LLVM none/O2; atol=1e-6, "
                         "rtol=1e-5)\n";
        return 0;
    } catch (const Diagnostic &error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "tensorforge: " << error.what() << '\n';
        return 1;
    }
}

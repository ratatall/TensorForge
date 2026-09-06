#include "tensorforge/benchmark.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/utsname.h>

namespace tensorforge {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
struct Samples {
    double median, checksum, p10, p90;
    std::size_t batch;
};
template <class F> Samples measure(std::size_t iterations, F call) {
    for (int warmup = 0; warmup < 5; ++warmup)
        call();
    // Calibrate a sample to >= 200 us to reduce timer overhead for tiny kernels.
    std::size_t batch = 1;
    while (batch < 65536) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < batch; ++i)
            call();
        if (elapsed(start) >= 200)
            break;
        batch *= 2;
    }
    std::vector<double> samples;
    double checksum = 0;
    for (std::size_t sample = 0; sample < iterations; ++sample) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < batch; ++i)
            checksum += call();
        samples.push_back(elapsed(start) / static_cast<double>(batch));
    }
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    return {(samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2),
            checksum, samples[(samples.size() - 1) / 10], samples[(samples.size() - 1) * 9 / 10],
            batch};
}
std::string quoted(const std::string &value) {
    std::string result = "\"";
    for (char c : value)
        result += c == '"' ? "\"\"" : std::string(1, c);
    return result + '"';
}
} // namespace
void benchmark(const Module &module, std::size_t iterations, std::uint32_t seed,
               const std::string &csvPath, const std::string &workload) {
    if (std::string(TF_BUILD_TYPE) != "Release")
        throw std::runtime_error("benchmarks require a Release build");
    if (iterations < 3 || iterations > 100000)
        throw std::invalid_argument("iterations must be in [3, 100000]");
    const auto inputs = generateInputs(module, seed);
    const auto oracle = interpret(module, inputs);
    struct Run {
        std::string name;
        bool project;
        LLVMOptimization level;
        Executable executable;
        double compile;
        Samples samples{};
    };
    std::vector<Run> runs;
    for (auto level : {LLVMOptimization::None, LLVMOptimization::O2}) {
        for (bool project : {false, true}) {
            const auto start = Clock::now();
            auto ir = module;
            if (project)
                PassManager().run(ir);
            Executable executable(ir, level);
            const double compile = elapsed(start);
            compare(executable.run(inputs), oracle);
            std::string name = project ? "jit-optimized" : "jit-unoptimized";
            if (level == LLVMOptimization::O2)
                name += "-O2";
            runs.push_back({name, project, level, std::move(executable), compile, {}});
        }
    }
    std::vector<const float *> pointers;
    for (const auto &input : inputs)
        pointers.push_back(input.data());
    const auto reference = measure(iterations, [&] {
        auto values = interpret(module, inputs);
        return static_cast<double>(values.front()) + values.back();
    });
    for (auto &run : runs) {
        Tensor output(oracle.size()), scratch(run.executable.stats().scratchElements);
        run.samples = measure(iterations, [&] {
            run.executable.invoke(pointers.data(), output.data(), scratch.data());
            return static_cast<double>(output.front()) + output.back();
        });
        compare(output, oracle);
    }
    struct utsname system{};
    if (uname(&system) != 0)
        throw std::runtime_error("uname failed");
    std::cout << "Workload: " << workload << "; elements=" << oracle.size() << "; seed=" << seed
              << "; samples=" << iterations << '\n'
              << TF_COMPILER << "; LLVM " << llvmVersion() << "; " << system.sysname << ' '
              << system.release << ' ' << system.machine
              << "; CPU=" << runs[0].executable.targetCPU() << "; " << TF_BUILD_TYPE
              << "; sanitizers=" << TF_SANITIZERS << '\n'
              << "Execution median (us); compile latency excludes parsing/semantic analysis.\n"
              << std::left << std::setw(22) << "mode" << std::setw(10) << "TF passes"
              << std::setw(8) << "LLVM" << std::setw(14) << "median_us" << std::setw(14)
              << "compile_us" << std::setw(14) << "lowered_loops" << std::setw(12) << "llvm_loops"
              << "scratch_bytes\n";
    auto row = [&](const char *name, const char *project, const char *level, Samples samples,
                   double compile, LoweringStats stats) {
        std::cout << std::left << std::setw(22) << name << std::setw(10) << project << std::setw(8)
                  << level << std::setw(14) << samples.median << std::setw(14) << compile
                  << std::setw(14) << stats.loops << std::setw(12) << stats.llvmLoops
                  << checkedFloatBytes(stats.scratchElements) << '\n';
    };
    row("interpreter", "reference", "n/a", reference, 0, {});
    for (const auto &run : runs)
        row(run.name.c_str(), run.project ? "on" : "off", optimizationName(run.level), run.samples,
            run.compile, run.executable.stats());
    std::cout << "Custom-pass speedup at LLVM=none: "
              << runs[0].samples.median / runs[1].samples.median
              << "x; all outputs verified.\nMachine-specific results. Interpreter includes "
                 "validation and allocations; JIT timings use prepared buffers.\n";
    if (!csvPath.empty()) {
        std::ofstream csv(csvPath);
        if (!csv)
            throw std::runtime_error("cannot write CSV: " + csvPath);
        csv << "workload,elements,seed,samples,batch,mode,project_opt,llvm_opt,median_us,p10_us,"
               "p90_us,compile_us,lowered_loops,llvm_loops,vector_instructions,scratch_bytes,"
               "checksum,verified,compiler,llvm,os,arch,cpu,build_type,sanitizers\n"
            << std::setprecision(12);
        auto write = [&](const char *mode, const char *project, const char *level, Samples samples,
                         double compile, LoweringStats stats) {
            csv << quoted(std::filesystem::path(workload).filename().string()) << ','
                << oracle.size() << ',' << seed << ',' << iterations << ',' << samples.batch << ','
                << mode << ',' << project << ',' << level << ',' << samples.median << ','
                << samples.p10 << ',' << samples.p90 << ',' << compile << ',' << stats.loops << ','
                << stats.llvmLoops << ',' << stats.vectorInstructions << ','
                << checkedFloatBytes(stats.scratchElements) << ',' << samples.checksum << ",true,"
                << quoted(TF_COMPILER) << ',' << llvmVersion() << ','
                << quoted(std::string(system.sysname) + ' ' + system.release) << ','
                << system.machine << ',' << quoted(runs[0].executable.targetCPU()) << ','
                << TF_BUILD_TYPE << ',' << TF_SANITIZERS << '\n';
        };
        write("interpreter", "reference", "n/a", reference, 0, {});
        for (const auto &run : runs)
            write(run.name.c_str(), run.project ? "on" : "off", optimizationName(run.level),
                  run.samples, run.compile, run.executable.stats());
        csv.flush();
        if (!csv)
            throw std::runtime_error("failed writing CSV: " + csvPath);
    }
}
} // namespace tensorforge

#pragma once
#include "tensorforge/codegen.h"
#include <cstdint>
#include <string>
namespace tensorforge {
void benchmark(const Module &module, std::size_t iterations, std::uint32_t seed,
               const std::string &csvPath, const std::string &workload);
}

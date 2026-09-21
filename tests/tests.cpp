#include "tensorforge/codegen.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

using namespace tensorforge;
namespace {
int assertions = 0;
void expect(bool condition, const std::string &message) {
    ++assertions;
    if (!condition)
        throw std::runtime_error(message);
}
Module lower(const std::string &text) {
    Source source{"test.tf", text};
    auto ast = parse(source);
    return analyzeAndLower(source, ast);
}
void rejects(const std::string &text, const std::string &message) {
    try {
        lower(text);
    } catch (const Diagnostic &error) {
        std::string diagnostic = error.what();
        expect(diagnostic.find("test.tf:") != std::string::npos &&
                   diagnostic.find('^') != std::string::npos,
               "diagnostic missing location/caret");
        expect(diagnostic.find(message) != std::string::npos,
               "unexpected diagnostic: " + diagnostic);
        return;
    }
    throw std::runtime_error("invalid program accepted: " + text);
}
std::size_t live(const Module &module, Opcode code) {
    std::size_t count = 0;
    for (const auto &op : module.operations)
        count += op.alive && op.opcode == code;
    return count;
}
void lexerTests() {
    auto tokens =
        lex({"tokens.tf",
             "// comment\ninput abc_2: tensor<17>; let x = .5e+1 * 2 + -3; return relu(x);"});
    expect(tokens.front().kind == TokenKind::Input, "keyword");
    expect(tokens[1].text == "abc_2", "identifier");
    expect(tokens.front().location.line == 2 && tokens.front().location.column == 1,
           "source location");
    expect(tokens[1].location.column == 7, "identifier column");
    expect(tokens.back().kind == TokenKind::End, "EOF");
    rejects("return @;", "invalid character");
    rejects("return 2e+;", "exponent digits");
    auto eof = lex({"empty", ""});
    expect(eof.size() == 1, "empty input lexing");
    auto f32 = lex({"tokens", "f32 ( ) = + * : ; < , >"});
    const std::vector<TokenKind> expected{
        TokenKind::F32,  TokenKind::LeftParen, TokenKind::RightParen, TokenKind::Equal,
        TokenKind::Plus, TokenKind::Star,      TokenKind::Colon,      TokenKind::Semicolon,
        TokenKind::Less, TokenKind::Comma,     TokenKind::Greater,    TokenKind::End};
    expect(f32.size() == expected.size(), "punctuation count");
    for (std::size_t i = 0; i < expected.size(); ++i)
        expect(f32[i].kind == expected[i], "punctuation kind");
}
void parserTests() {
    Source source{"test.tf", "return 2 + 3 * 4;"};
    auto ast = parse(source);
    const auto &add = std::get<Binary>(ast.result.expression->node);
    expect(add.op == '+' && std::get<Binary>(add.right->node).op == '*', "operator precedence");
    expect(interpret(lower("return (2 + 3) * 4;"), {}) == Tensor{20}, "parentheses");
    expect(interpret(lower("return relu(relu(-2));"), {}) == Tensor{0}, "nested relu");
    expect(printAST(ast) == "Program @1:1\n  Return @1:1\n    Binary + @1:8\n      Number 2 @1:8\n "
                            "     Binary * @1:12\n        Number 3 @1:12\n        Number 4 @1:16\n",
           "deterministic AST");
    rejects("return 2", "expected ';'");
    rejects("return );", "expected expression");
    rejects("return 1; return 2;", "end of file");
    rejects("let x = 2;", "final return");
    rejects("return -relu(1);", "numeric literal");
    rejects("return 1e100;", "finite f32 range");
    rejects("return " + std::string(129, '(') + "1" + std::string(129, ')') + ";",
            "nesting exceeds");
}
void semanticTests() {
    rejects("return missing;", "undeclared");
    rejects("input A: f32; let A = 2; return A;", "duplicate");
    rejects("let A = 2; input A: f32; return A;", "duplicate");
    rejects("let A = A + 1; return A;", "undeclared");
    rejects("input A: tensor<4>; input B: tensor<5>; return A + B;", "shape mismatch");
    rejects("input A: tensor<2,3>; input B: tensor<2,2>; return A + B;", "shape mismatch");
    for (auto size : {"0", "1.5", "1e2", "16777217", "999999999999999999999999"})
        rejects("input A: tensor<" + std::string(size) + ">; return A;", "tensor extent");
    rejects("input A: tensor<-1>; return A;", "positive integer");
    rejects("input A: tensor<4096,4097>; return A;", "shape exceeds");
    rejects("input A: tensor<1,1,1,1,1,1,1,1,1>; return A;", "rank exceeds");
    for (const auto &expression : {"A + B", "A * 2", "2 + A", "relu(A)"}) {
        auto module = lower("input A: tensor<1>; input B: tensor<1>; return " +
                            std::string(expression) + ";");
        expect(module.operations[module.result].type == Type{1}, "tensor<1> distinct from scalar");
    }
    auto scalar = lower("input s: f32; return s * 2;");
    expect(scalar.operations[scalar.result].type.scalar(), "scalar input type");
    auto matrix = lower("input A: tensor<2,3>; input B: tensor<3>; return A+B;");
    expect(matrix.operations[matrix.result].type == Type({2, 3}),
           "trailing-dimension broadcasting");
    auto outer = lower("input A: tensor<2,1>; input B: tensor<1,3>; return A*B;");
    expect(outer.operations[outer.result].type == Type({2, 3}), "two-axis broadcasting");
}
void irTests() {
    auto module = lower("input A: tensor<4>; return A * 2;");
    expect(printIR(module) ==
               "module {\n  %0 = input \"A\" [argument 0] : tensor<4xf32>\n  %1 = constant 2 : "
               "f32\n  %2 = multiply %0 %1 : tensor<4xf32>\n  return %2\n}\n",
           "IR snapshot");
    auto broken = module;
    broken.operations[2].operands[0] = 2;
    bool caught = false;
    try {
        validateIR(broken);
    } catch (const std::logic_error &) {
        caught = true;
    }
    expect(caught, "IR rejects forward references");
    expect(module.inputs[0].name == "A", "ABI input order");
}
void passTests() {
    auto module =
        lower("input A: tensor<4>; let scale = 2 * 4; let dead = A * 3; return relu(A * scale);");
    auto original = module;
    expect(constantFold(module), "fold changed");
    expect(module.operations[3].opcode == Opcode::Constant && module.operations[3].constant == 8,
           "folded 8");
    expect(!constantFold(module), "fold idempotent");
    expect(eliminateDeadCode(module), "DCE changed");
    expect(live(module, Opcode::Multiply) == 1, "only live tensor multiply remains");
    expect(fuseElementwise(module), "fusion changed");
    expect(module.fusedRegion.size() == 2, "fusion schedule");
    expect(!fuseElementwise(module), "fusion idempotent");
    expect(printIR(module).find("fused_elementwise") != std::string::npos, "visible fusion");
    compare(interpret(module, generateInputs(module, 5)),
            interpret(original, generateInputs(original, 5)));
    auto unusedInput = lower("input unused: tensor<9>; input live: tensor<4>; return live;");
    eliminateDeadCode(unusedInput);
    expect(unusedInput.inputs.size() == 2 && unusedInput.operations[1].inputIndex == 1,
           "DCE preserves ABI indices");
    auto scalar = lower("return relu(2 * -4) + 1;");
    PassManager().run(scalar);
    expect(live(scalar, Opcode::Constant) == 1 && scalar.operations[scalar.result].constant == 1,
           "transitive fold and DCE");
    std::ostringstream trace;
    PassManager().run(original, &trace);
    expect(trace.str().find("after elementwise-fusion") != std::string::npos, "pass trace");
}
void checkBoth(const Module &module, const Inputs &inputs) {
    const auto oracle = interpret(module, inputs);
    auto optimized = module;
    PassManager().run(optimized);
    for (auto level : {LLVMOptimization::None, LLVMOptimization::O2}) {
        compare(Executable(module, level).run(inputs), oracle);
        compare(Executable(optimized, level).run(inputs), oracle);
        assertions += 2;
    }
}
void codegenTests() {
    auto module = lower("input A: tensor<17>; input B: tensor<17>; return relu(A * 2 + B);");
    Executable plain(module);
    expect(plain.stats().loops == 3 && plain.stats().scratchElements == 34,
           "unoptimized structure");
    PassManager().run(module);
    Executable fused(module);
    expect(fused.stats().loops == 1 && fused.stats().scratchElements == 0, "fused structure");
    expect(fused.llvmIR().find("fcmp ogt") != std::string::npos &&
               fused.llvmIR().find("fmul") != std::string::npos,
           "LLVM arithmetic");
    auto countLoops = [](const std::string &text) {
        std::size_t count = 0, pos = 0;
        while ((pos = text.find("phi i64", pos)) != std::string::npos) {
            ++count;
            ++pos;
        }
        return count;
    };
    expect(countLoops(plain.llvmIR()) == 3 && countLoops(fused.llvmIR()) == 1,
           "actual LLVM loop PHIs match lowering schedule");
    auto broadcast = lower("input x: tensor<4>; input s: f32; return relu(2 * x + s);");
    Inputs known{{-2, -0.5f, 0, 3}, {1}};
    expect(interpret(broadcast, known) == Tensor({0, 0, 1, 7}), "independent broadcast oracle");
    checkBoth(broadcast, known);
    checkBoth(lower("input dead: tensor<2>; input x: f32; let unused = dead * 2; return x;"),
              {{9, 8}, {3}});
    checkBoth(lower("input dead: tensor<2>; input x: tensor<1>; return x;"), {{9, 8}, {3}});
    checkBoth(lower("return 2 + 3 * 4;"), {});
    checkBoth(lower("return relu((3e38 * 2) * 0);"), {});
    checkBoth(lower("input x: f32; return relu(x);"), {{-1}});
    checkBoth(lower("input x: tensor<1>; return x;"), {{5}});
    const float nan = std::numeric_limits<float>::quiet_NaN(),
                inf = std::numeric_limits<float>::infinity();
    auto relu = lower("input x: tensor<7>; return relu(x);");
    Inputs special{{nan, -0.0f, 0.0f, -inf, inf, -1e-30f, 1e-30f}};
    checkBoth(relu, special);
    for (auto level : {LLVMOptimization::None, LLVMOptimization::O2}) {
        for (bool project : {false, true}) {
            auto ir = relu;
            if (project)
                PassManager().run(ir);
            auto result = Executable(ir, level).run(special);
            expect(result[0] == 0 && !std::signbit(result[0]) && !std::signbit(result[1]) &&
                       !std::signbit(result[2]),
                   "ReLU NaN and signed zero policy in every mode");
        }
    }
    auto nanExpr = lower("input x: tensor<2>; return x * 0;");
    checkBoth(nanExpr, {{nan, inf}});
    auto simplify = lower("input x: tensor<17>; return x*1;");
    auto before = Executable(simplify, LLVMOptimization::None).llvmIR();
    auto after = Executable(simplify, LLVMOptimization::O2).llvmIR();
    expect(before.find("fmul") != std::string::npos && after.find("fmul") == std::string::npos,
           "O2 runs and emitted IR reflects it");
    checkBoth(simplify, generateInputs(simplify, 3));
    bool invalid = false;
    try {
        plain.run({});
    } catch (const std::invalid_argument &) {
        invalid = true;
    }
    expect(invalid, "runtime input check");
}
template <class Exception, class F> void throwsWith(F action, const std::string &text) {
    try {
        action();
    } catch (const Exception &error) {
        expect(std::string(error.what()).find(text) != std::string::npos,
               "wrong exception: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected exception containing: " + text);
}
void resourceTests() {
    rejects("", "final return");
    rejects("// eof", "final return");
    rejects("input", "input name");
    rejects("return 1e-;", "exponent digits");
    rejects("return .;", "invalid character");
    rejects("return \xc3\xa9;", "invalid character");
    rejects(std::string("return ") + char(0) + ";", "invalid character");
    auto tokens = lex({"crlf.tf", " \t// comment\r\nreturn 1; // final"});
    expect(tokens[0].location.line == 2 && tokens[0].location.column == 1, "CRLF/comment location");
    throwsWith<Diagnostic>(
        [] { lower("input x: f32;\nreturn missing;"); },
        "test.tf:2:8: error: undeclared identifier 'missing'\nreturn missing;\n       ^");
    expect(lower("input x: tensor<16777216>; return x;").inputs[0].type.shape ==
               std::vector<std::size_t>{MaxTensorExtent},
           "max legal extent frontend");
    rejects("input x: tensor<18446744073709551616>; return x;", "tensor extent");
    lower("return " + std::string(127, '(') + "1" + std::string(127, ')') + ";");
    rejects("return " + std::string(128, '(') + "1" + std::string(128, ')') + ";",
            "nesting exceeds");
    std::string expressions;
    for (int i = 0; i < 2047; ++i)
        expressions += "let x" + std::to_string(i) + " = 1;";
    lower(expressions + "return 1;");
    rejects(expressions + "return 1+1;", "2048-expression");
    std::string tokenLimit;
    for (int i = 0; i < 16384; ++i)
        tokenLimit += "+ ";
    expect(lex({"tokens", tokenLimit}).size() == 16385, "max tokens plus EOF");
    throwsWith<Diagnostic>([&] { lex({"tokens", tokenLimit + "+"}); }, "16384-token");
    throwsWith<std::overflow_error>(
        [] { checkedFloatBytes(std::numeric_limits<std::size_t>::max()); }, "overflow");
    expect(checkedFloatBytes(MaxBufferElements) == 1073741824, "exact byte count");
    expect(addBufferElements(MaxBufferElements - 1, 1) == MaxBufferElements, "budget boundary");
    throwsWith<std::length_error>([] { addBufferElements(MaxBufferElements, 1); }, "1 GiB");
    auto base = lower("input x: f32; return x;");
    auto invalid = base;
    invalid.operations[0].opcode = static_cast<Opcode>(99);
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "unknown opcode");
    invalid = base;
    invalid.inputs[0].type.shape = {std::numeric_limits<std::size_t>::max()};
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "shape exceeds");
    invalid = base;
    invalid.result = 100;
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "return value");
    invalid = base;
    invalid.operations[0].alive = false;
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "return value is dead");
    invalid = base;
    invalid.operations[0].inputIndex = 1;
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "input index");
    invalid = lower("input x: tensor<4>; return x*2;");
    invalid.operations.back().type = {3};
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "incorrect result type");
    invalid = lower("input x: tensor<4>; return x*2;");
    invalid.operations.back().operands.clear();
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "operand count");
    invalid = lower("input x: tensor<4>; return relu(x*2);");
    PassManager().run(invalid);
    invalid.fusedRegion.pop_back();
    throwsWith<std::logic_error>([&] { validateIR(invalid); }, "fusion schedule");
    auto scheduled = lower("input x: tensor<4>; return relu(x);");
    PassManager().run(scheduled);
    expect(eliminateDeadCode(scheduled) && scheduled.fusedRegion.empty(),
           "DCE reports schedule invalidation");
    auto trailing =
        lower("input x: tensor<4>; let result=relu(x); let dead=result*2; return result;");
    expect(!fuseElementwise(trailing) && trailing.fusedRegion.empty(),
           "fusion refuses unscheduled trailing dead computation");
    std::string bigInputs;
    for (int i = 0; i < 17; ++i)
        bigInputs += "input x" + std::to_string(i) + ": tensor<16777216>;";
    auto large = lower(bigInputs + "return x0;");
    throwsWith<std::length_error>([&] { generateInputs(large, 0); }, "1 GiB");
    std::string chain = "input x: tensor<16777216>; let v0=x*2;";
    for (int i = 1; i < 18; ++i)
        chain += "let v" + std::to_string(i) + "=v" + std::to_string(i - 1) + "*2;";
    auto scratch = lower(chain + "return v17;");
    throwsWith<std::length_error>([&] { Executable executable(scratch); }, "1 GiB");
}
void runtimeTests() {
    auto m = lower("input a: tensor<4>; input b: f32; return a*b+b;");
    expect(generateInputs(m, 42) == generateInputs(m, 42), "same seed same inputs");
    expect(generateInputs(m, 42) != generateInputs(m, 43), "different seeds");
    Inputs known{{-2, 0, 1, 2}, {2}};
    expect(interpret(m, known) == Tensor({-2, 2, 4, 6}), "runtime expected values and order");
    auto matrix = lower("input A: tensor<2,3>; input row: tensor<3>; return A+row;");
    Inputs matrixInputs{{1, 2, 3, 4, 5, 6}, {10, 20, 30}};
    expect(interpret(matrix, matrixInputs) == Tensor({11, 22, 33, 14, 25, 36}),
           "row broadcast oracle");
    checkBoth(matrix, matrixInputs);
    auto mixed =
        lower("input A: tensor<2,3>; input row: tensor<3>; let scaled=row*2; return A+scaled;");
    auto mixedOptimized = mixed;
    PassManager().run(mixedOptimized);
    expect(mixedOptimized.fusedRegion.empty(), "mixed-shape graph stays unfused");
    checkBoth(mixed, matrixInputs);
    auto outer = lower("input column: tensor<2,1>; input row: tensor<1,3>; return column*row;");
    Inputs outerInputs{{2, 4}, {10, 20, 30}};
    expect(interpret(outer, outerInputs) == Tensor({20, 40, 60, 40, 80, 120}),
           "multi-axis broadcast oracle");
    checkBoth(outer, outerInputs);
    throwsWith<std::invalid_argument>([&] { interpret(m, {}); }, "input count");
    throwsWith<std::invalid_argument>([&] { Executable(m).run({{1}, {2}}); }, "input shape");
    throwsWith<std::invalid_argument>([&] { interpret(m, {{1, 2, 3, 4}, {}}); }, "input shape");
    compare({1.000005f}, {1});
    compare({0.0000005f}, {0});
    throwsWith<std::runtime_error>([] { compare({1}, {2}); }, "element 0: actual=1, expected=2");
    throwsWith<std::runtime_error>([] { compare({0.000002f}, {0}); }, "element 0");
    throwsWith<std::runtime_error>([] { compare({}, {1}); }, "shape mismatch");
    const float nan = std::numeric_limits<float>::quiet_NaN(),
                inf = std::numeric_limits<float>::infinity();
    compare({nan, inf, -inf, -0.0f}, {nan, inf, -inf, 0.0f});
    throwsWith<std::runtime_error>([&] { compare({nan}, {0}); }, "element 0");
    throwsWith<std::runtime_error>([&] { compare({inf}, {-inf}); }, "element 0");
    throwsWith<std::invalid_argument>([&] { compare({1}, {1}, nan); }, "tolerances");
    throwsWith<std::invalid_argument>([] { compare({1}, {1}, -1); }, "tolerances");
    auto dead = lower("input a: tensor<4>; let unused=a*9; return 2+a;");
    checkBoth(dead, {{-1, 0, 1, 2}});
}
void differentialTests() {
    std::mt19937 random(2028);
    for (auto size : {1, 2, 7, 31, 256, 1024}) {
        const std::string prefix = "input A: tensor<" + std::to_string(size) +
                                   ">; input B: tensor<" + std::to_string(size) +
                                   ">; input s: f32; ";
        std::vector<std::string> expressions{"A + B", "2 * A + s", "relu(A * 0.5 + B)",
                                             "relu(A) * relu(B)", "(A + B) * (s + 2)"};
        // Deterministic random expression trees test more than fixed examples.
        std::function<std::string(int)> expression = [&](int depth) -> std::string {
            if (depth == 0) {
                const char *leaves[]{"A", "B", "s", "-0.25"};
                return leaves[random() % 4];
            }
            const auto choice = random() % 3;
            if (choice == 0)
                return "relu(" + expression(depth - 1) + ")";
            auto left = expression(depth - 1), right = expression(depth - 1);
            return "(" + left + (choice == 1 ? " + " : " * ") + right + ")";
        };
        for (int i = 0; i < 5; ++i)
            expressions.push_back(expression(3));
        for (const auto &expr : expressions) {
            const auto module = lower(prefix + "return " + expr + ";");
            // Compile once, exercise several seeds per compiled graph.
            auto optimized = module;
            PassManager().run(optimized);
            for (auto level : {LLVMOptimization::None, LLVMOptimization::O2}) {
                Executable plain(module, level), fused(optimized, level);
                for (auto seed : {0U, 1U, 42U, 2028U}) {
                    auto inputs = generateInputs(module, seed);
                    auto expected = interpret(module, inputs);
                    compare(plain.run(inputs), expected);
                    compare(fused.run(inputs), expected);
                    assertions += 2;
                }
            }
        }
        checkBoth(lower(prefix + "let v = A * s; return relu(v * v + v + B);"),
                  generateInputs(lower(prefix + "return A;"), 19));
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("expected test group");
        const std::string group = argv[1];
        if (group == "lexer")
            lexerTests();
        else if (group == "parser")
            parserTests();
        else if (group == "semantic")
            semanticTests();
        else if (group == "ir")
            irTests();
        else if (group == "passes")
            passTests();
        else if (group == "codegen")
            codegenTests();
        else if (group == "resources")
            resourceTests();
        else if (group == "runtime")
            runtimeTests();
        else if (group == "assertion-probe")
            expect(false, "intentional assertion probe");
        else if (group == "differential")
            differentialTests();
        else
            throw std::invalid_argument("unknown test group");
        std::cout << group << ": " << assertions << " assertions passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

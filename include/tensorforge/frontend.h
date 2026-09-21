#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tensorforge {
inline constexpr std::size_t MaxTensorExtent = 16777216;
inline constexpr std::size_t MaxTensorRank = 8;
struct Location {
    std::size_t offset = 0, line = 1, column = 1;
};
struct Source {
    std::string filename, text;
    [[noreturn]] void fail(Location where, const std::string &message) const;
};
class Diagnostic : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class TokenKind {
    End,
    Identifier,
    Number,
    Input,
    Let,
    Return,
    F32,
    Tensor,
    Relu,
    Colon,
    Semicolon,
    Equal,
    Less,
    Greater,
    Plus,
    Star,
    Minus,
    Comma,
    LeftParen,
    RightParen
};
struct Token {
    TokenKind kind;
    std::string text;
    Location location;
};
std::vector<Token> lex(const Source &source);

struct Type {
    // An empty shape denotes a scalar; tensor<1> remains a distinct type.
    std::vector<std::size_t> shape;
    Type() = default;
    Type(std::initializer_list<std::size_t> dimensions) : shape(dimensions) {}
    bool scalar() const { return shape.empty(); }
    std::size_t elements() const;
    std::string str() const;
    bool operator==(const Type &) const = default;
};
struct Expr;
using ExprPtr = std::unique_ptr<Expr>;
struct Identifier {
    std::string name;
};
struct Number {
    float value;
};
struct Binary {
    char op;
    ExprPtr left, right;
};
struct Relu {
    ExprPtr argument;
};
struct Expr {
    Location location;
    std::variant<Identifier, Number, Binary, Relu> node;
};
struct InputDecl {
    Location location;
    std::string name;
    Type type;
};
struct LetDecl {
    Location location;
    std::string name;
    ExprPtr expression;
};
struct ReturnStmt {
    Location location;
    ExprPtr expression;
};
using Declaration = std::variant<InputDecl, LetDecl>;
struct Program {
    Location location;
    std::vector<Declaration> declarations;
    ReturnStmt result;
};
Program parse(const Source &source);
std::string printAST(const Program &program);
} // namespace tensorforge

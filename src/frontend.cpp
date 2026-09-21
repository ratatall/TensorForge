#include "tensorforge/frontend.h"
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <unordered_map>

namespace tensorforge {
[[noreturn]] void Source::fail(Location where, const std::string &message) const {
    const auto start = where.offset == 0 ? std::string::npos : text.rfind('\n', where.offset - 1);
    const auto begin = start == std::string::npos ? 0 : start + 1;
    const auto end = text.find('\n', begin);
    std::string line, caret;
    for (std::size_t i = begin; i < text.size() && i != end; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        std::string rendered;
        if (c == '\t' || (c >= 32 && c < 127))
            rendered = static_cast<char>(c);
        else {
            constexpr char hex[] = "0123456789abcdef";
            rendered = "\\x";
            rendered += hex[c >> 4];
            rendered += hex[c & 15];
        }
        line += rendered;
        if (i < where.offset)
            caret += c == '\t' ? "\t" : std::string(rendered.size(), ' ');
    }
    std::ostringstream out;
    out << filename << ':' << where.line << ':' << where.column << ": error: " << message << '\n'
        << line << '\n'
        << caret << '^';
    throw Diagnostic(out.str());
}
std::string Type::str() const {
    if (scalar())
        return "f32";
    std::ostringstream out;
    out << "tensor<";
    for (auto dimension : shape)
        out << dimension << 'x';
    out << "f32>";
    return out.str();
}
std::size_t Type::elements() const {
    std::size_t total = 1;
    for (auto dimension : shape) {
        if (dimension == 0 || total > MaxTensorExtent / dimension)
            throw std::length_error("tensor shape exceeds 16777216-element limit");
        total *= dimension;
    }
    return total;
}
namespace {
bool digit(char c) {
    return c >= '0' && c <= '9';
}
bool alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
} // namespace
std::vector<Token> lex(const Source &source) {
    std::vector<Token> tokens;
    Location pos;
    auto peek = [&](std::size_t delta = 0) {
        return pos.offset + delta < source.text.size() ? source.text[pos.offset + delta] : '\0';
    };
    auto advance = [&] {
        if (peek() == '\n') {
            ++pos.line;
            pos.column = 1;
        } else
            ++pos.column;
        ++pos.offset;
    };
    const std::unordered_map<std::string, TokenKind> keywords = {
        {"input", TokenKind::Input}, {"let", TokenKind::Let},       {"return", TokenKind::Return},
        {"f32", TokenKind::F32},     {"tensor", TokenKind::Tensor}, {"relu", TokenKind::Relu}};
    while (pos.offset < source.text.size()) {
        const char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
            continue;
        }
        if (c == '/' && peek(1) == '/') {
            while (pos.offset < source.text.size() && peek() != '\n')
                advance();
            continue;
        }
        const auto start = pos;
        TokenKind kind;
        if (alpha(c)) {
            while (alpha(peek()) || digit(peek()))
                advance();
            auto word = source.text.substr(start.offset, pos.offset - start.offset);
            auto found = keywords.find(word);
            kind = found == keywords.end() ? TokenKind::Identifier : found->second;
        } else if (digit(c) || (c == '.' && digit(peek(1)))) {
            while (digit(peek()))
                advance();
            if (peek() == '.') {
                advance();
                while (digit(peek()))
                    advance();
            }
            if (peek() == 'e' || peek() == 'E') {
                advance();
                if (peek() == '+' || peek() == '-')
                    advance();
                if (!digit(peek()))
                    source.fail(pos, "expected exponent digits");
                while (digit(peek()))
                    advance();
            }
            kind = TokenKind::Number;
        } else {
            switch (c) {
            case ':':
                kind = TokenKind::Colon;
                break;
            case ';':
                kind = TokenKind::Semicolon;
                break;
            case '=':
                kind = TokenKind::Equal;
                break;
            case '<':
                kind = TokenKind::Less;
                break;
            case '>':
                kind = TokenKind::Greater;
                break;
            case '+':
                kind = TokenKind::Plus;
                break;
            case '*':
                kind = TokenKind::Star;
                break;
            case '-':
                kind = TokenKind::Minus;
                break;
            case ',':
                kind = TokenKind::Comma;
                break;
            case '(':
                kind = TokenKind::LeftParen;
                break;
            case ')':
                kind = TokenKind::RightParen;
                break;
            default:
                source.fail(pos, "invalid character");
            }
            advance();
        }
        tokens.push_back(
            {kind, source.text.substr(start.offset, pos.offset - start.offset), start});
        if (tokens.size() > 16384)
            source.fail(start, "program exceeds 16384-token limit");
    }
    tokens.push_back({TokenKind::End, "", pos});
    return tokens;
}
namespace {
class Parser {
    const Source &source_;
    std::vector<Token> tokens_;
    std::size_t cursor_ = 0, depth_ = 0, expressions_ = 0;
    const Token &current() const { return tokens_.at(cursor_); }
    bool accept(TokenKind kind) {
        if (current().kind != kind)
            return false;
        ++cursor_;
        return true;
    }
    Token require(TokenKind kind, const std::string &what) {
        if (current().kind != kind)
            source_.fail(current().location, "expected " + what);
        return tokens_.at(cursor_++);
    }
    template <class T> ExprPtr make(Location loc, T node) {
        if (++expressions_ > 2048)
            source_.fail(loc, "program exceeds 2048-expression limit");
        return std::make_unique<Expr>(Expr{loc, std::move(node)});
    }
    ExprPtr primary() {
        const auto token = current();
        if (accept(TokenKind::Identifier))
            return make(token.location, Identifier{token.text});
        bool negative = accept(TokenKind::Minus);
        if (current().kind == TokenKind::Number) {
            const auto number = require(TokenKind::Number, "numeric literal");
            float value = 0;
            // Apple's system libc++ lacks floating-point from_chars. A classic
            // locale stream keeps decimal syntax independent of the host locale.
            std::istringstream literal(number.text);
            literal.imbue(std::locale::classic());
            literal >> value;
            if (literal.fail() || !std::isfinite(value))
                source_.fail(number.location, "literal is outside finite f32 range");
            return make(token.location, Number{negative ? -value : value});
        }
        if (negative)
            source_.fail(current().location, "expected numeric literal after '-'");
        if (accept(TokenKind::Relu)) {
            require(TokenKind::LeftParen, "'('");
            auto arg = expression();
            require(TokenKind::RightParen, "')'");
            return make(token.location, Relu{std::move(arg)});
        }
        if (accept(TokenKind::LeftParen)) {
            auto arg = expression();
            require(TokenKind::RightParen, "')'");
            return arg;
        }
        source_.fail(token.location, "expected expression");
    }
    ExprPtr multiply() {
        auto left = primary();
        while (accept(TokenKind::Star)) {
            auto loc = left->location;
            left = make(loc, Binary{'*', std::move(left), primary()});
        }
        return left;
    }
    ExprPtr expression() {
        if (++depth_ > 128)
            source_.fail(current().location, "expression nesting exceeds 128");
        auto left = multiply();
        while (accept(TokenKind::Plus)) {
            auto loc = left->location;
            left = make(loc, Binary{'+', std::move(left), multiply()});
        }
        --depth_;
        return left;
    }
    Type type() {
        if (accept(TokenKind::F32))
            return {};
        require(TokenKind::Tensor, "f32 or tensor type");
        require(TokenKind::Less, "'<'");
        Type result;
        do {
            const auto extent = require(TokenKind::Number, "positive integer tensor extent");
            std::size_t size = 0;
            auto parsed =
                std::from_chars(extent.text.data(), extent.text.data() + extent.text.size(), size);
            if (parsed.ec != std::errc{} || parsed.ptr != extent.text.data() + extent.text.size() ||
                size == 0 || size > MaxTensorExtent)
                source_.fail(extent.location, "tensor extent must be an integer in [1, 16777216]");
            result.shape.push_back(size);
            if (result.shape.size() > MaxTensorRank)
                source_.fail(extent.location, "tensor rank exceeds 8");
        } while (accept(TokenKind::Comma));
        require(TokenKind::Greater, "'>'");
        try {
            (void)result.elements();
        } catch (const std::length_error &) {
            source_.fail(current().location, "tensor shape exceeds 16777216 elements");
        }
        return result;
    }

  public:
    explicit Parser(const Source &source) : source_(source), tokens_(lex(source)) {}
    Program run() {
        Program program{current().location, {}, {current().location, nullptr}};
        while (current().kind != TokenKind::Return) {
            const auto loc = current().location;
            if (accept(TokenKind::Input)) {
                const auto name = require(TokenKind::Identifier, "input name");
                require(TokenKind::Colon, "':'");
                auto declared = type();
                require(TokenKind::Semicolon, "';'");
                program.declarations.emplace_back(InputDecl{loc, name.text, declared});
            } else if (accept(TokenKind::Let)) {
                const auto name = require(TokenKind::Identifier, "binding name");
                require(TokenKind::Equal, "'='");
                auto expr = expression();
                require(TokenKind::Semicolon, "';'");
                program.declarations.emplace_back(LetDecl{loc, name.text, std::move(expr)});
            } else
                source_.fail(loc, "expected input, let, or final return statement");
        }
        auto loc = require(TokenKind::Return, "return").location;
        auto result = expression();
        require(TokenKind::Semicolon, "';'");
        require(TokenKind::End, "end of file after final return");
        program.result = {loc, std::move(result)};
        return program;
    }
};
void dumpExpr(std::ostream &out, const Expr &expr, std::size_t indent) {
    out << std::string(indent, ' ');
    std::visit(
        [&](const auto &node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, Identifier>)
                out << "Identifier " << node.name;
            else if constexpr (std::is_same_v<T, Number>)
                out << "Number " << std::setprecision(9) << node.value;
            else if constexpr (std::is_same_v<T, Binary>)
                out << "Binary " << node.op;
            else
                out << "ReLU";
            out << " @" << expr.location.line << ':' << expr.location.column << '\n';
            if constexpr (std::is_same_v<T, Binary>) {
                dumpExpr(out, *node.left, indent + 2);
                dumpExpr(out, *node.right, indent + 2);
            } else if constexpr (std::is_same_v<T, Relu>)
                dumpExpr(out, *node.argument, indent + 2);
        },
        expr.node);
}
} // namespace
Program parse(const Source &source) {
    return Parser(source).run();
}
std::string printAST(const Program &program) {
    std::ostringstream out;
    out << "Program @" << program.location.line << ':' << program.location.column << '\n';
    for (const auto &decl : program.declarations)
        std::visit(
            [&](const auto &node) {
                using T = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<T, InputDecl>)
                    out << "  Input " << node.name << " : " << node.type.str();
                else
                    out << "  Let " << node.name;
                out << " @" << node.location.line << ':' << node.location.column << '\n';
                if constexpr (std::is_same_v<T, LetDecl>)
                    dumpExpr(out, *node.expression, 4);
            },
            decl);
    out << "  Return @" << program.result.location.line << ':' << program.result.location.column
        << '\n';
    dumpExpr(out, *program.result.expression, 4);
    return out.str();
}
} // namespace tensorforge

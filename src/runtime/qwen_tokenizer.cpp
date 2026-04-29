#include "runtime/qwen_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_infer {
namespace {

constexpr int64_t kImStartTokenId = 151644;
constexpr int64_t kEosTokenId = 151645;

void AppendUtf8(uint32_t codepoint, std::string* out) {
    if (codepoint <= 0x7F) {
        out->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out->push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out->push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

bool NextUtf8Codepoint(const std::string& text, std::size_t* offset, uint32_t* codepoint) {
    if (*offset >= text.size()) {
        return false;
    }
    const auto c0 = static_cast<unsigned char>(text[*offset]);
    if (c0 < 0x80) {
        *codepoint = c0;
        ++(*offset);
        return true;
    }
    int len = 0;
    uint32_t value = 0;
    if ((c0 & 0xE0) == 0xC0) {
        len = 2;
        value = c0 & 0x1F;
    } else if ((c0 & 0xF0) == 0xE0) {
        len = 3;
        value = c0 & 0x0F;
    } else if ((c0 & 0xF8) == 0xF0) {
        len = 4;
        value = c0 & 0x07;
    } else {
        ++(*offset);
        return false;
    }
    if (*offset + static_cast<std::size_t>(len) > text.size()) {
        *offset = text.size();
        return false;
    }
    for (int i = 1; i < len; ++i) {
        const auto ci = static_cast<unsigned char>(text[*offset + static_cast<std::size_t>(i)]);
        if ((ci & 0xC0) != 0x80) {
            ++(*offset);
            return false;
        }
        value = (value << 6) | (ci & 0x3F);
    }
    *offset += static_cast<std::size_t>(len);
    *codepoint = value;
    return true;
}

std::string Trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::unordered_map<uint32_t, unsigned char> BuildByteDecoder() {
    std::vector<int> bytes;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        bytes.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bytes.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bytes.push_back(b);
    }

    std::vector<int> chars = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            chars.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<uint32_t, unsigned char> decoder;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        decoder[static_cast<uint32_t>(chars[i])] = static_cast<unsigned char>(bytes[i]);
    }
    return decoder;
}

std::unordered_map<unsigned char, std::string> BuildByteEncoder() {
    std::vector<int> bytes;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        bytes.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bytes.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bytes.push_back(b);
    }

    std::vector<int> chars = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            chars.push_back(256 + n);
            ++n;
        }
    }

    std::unordered_map<unsigned char, std::string> encoder;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        std::string encoded;
        AppendUtf8(static_cast<uint32_t>(chars[i]), &encoded);
        encoder[static_cast<unsigned char>(bytes[i])] = std::move(encoded);
    }
    return encoder;
}

std::string DecodeJsonString(const std::string& text, std::size_t* offset) {
    if (*offset >= text.size() || text[*offset] != '"') {
        throw std::runtime_error("expected JSON string");
    }
    ++(*offset);
    std::string out;
    while (*offset < text.size()) {
        const char c = text[(*offset)++];
        if (c == '"') {
            return out;
        }
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (*offset >= text.size()) {
            throw std::runtime_error("bad JSON escape");
        }
        const char esc = text[(*offset)++];
        switch (esc) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                if (*offset + 4 > text.size()) {
                    throw std::runtime_error("bad JSON unicode escape");
                }
                const std::string hex = text.substr(*offset, 4);
                *offset += 4;
                uint32_t cp = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
                if (cp >= 0xD800 && cp <= 0xDBFF && *offset + 6 <= text.size()
                    && text[*offset] == '\\' && text[*offset + 1] == 'u') {
                    *offset += 2;
                    const std::string low_hex = text.substr(*offset, 4);
                    *offset += 4;
                    const uint32_t low = static_cast<uint32_t>(std::stoul(low_hex, nullptr, 16));
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                }
                AppendUtf8(cp, &out);
                break;
            }
            default:
                throw std::runtime_error("unsupported JSON escape");
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

bool IsSpecialToken(const std::string& token) {
    return token.size() >= 2 && token.front() == '<' && token.back() == '>';
}

bool IsAsciiLetter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAsciiDigit(unsigned char c) {
    return c >= '0' && c <= '9';
}

bool IsWhitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool IsUtf8Continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

std::string PairKey(const std::string& first, const std::string& second) {
    return first + '\x1F' + second;
}

}  // namespace

struct QwenTokenizer::Impl {
    Impl(const std::string& vocab_path, const std::string& merges_path) {
        byte_encoder = BuildByteEncoder();
        byte_decoder = BuildByteDecoder();
        LoadVocab(vocab_path);
        LoadMerges(merges_path);
    }

    std::string Decode(int64_t token_id) const {
        auto it = id_to_token.find(token_id);
        if (it == id_to_token.end() || IsSpecialToken(it->second)) {
            return "";
        }

        std::string bytes;
        std::size_t offset = 0;
        while (offset < it->second.size()) {
            const std::size_t before = offset;
            uint32_t cp = 0;
            if (!NextUtf8Codepoint(it->second, &offset, &cp)) {
                bytes.push_back(it->second[before]);
                continue;
            }
            auto byte_it = byte_decoder.find(cp);
            if (byte_it != byte_decoder.end()) {
                bytes.push_back(static_cast<char>(byte_it->second));
            } else {
                AppendUtf8(cp, &bytes);
            }
        }
        return bytes;
    }

    std::vector<int64_t> EncodeChat(const std::string& system, const std::string& prompt) const {
        std::vector<int64_t> ids;
        AppendId(kImStartTokenId, &ids);
        AppendEncoded("system\n" + system, &ids);
        AppendId(kEosTokenId, &ids);
        AppendEncoded("\n", &ids);
        AppendId(kImStartTokenId, &ids);
        AppendEncoded("user\n" + prompt, &ids);
        AppendId(kEosTokenId, &ids);
        AppendEncoded("\n", &ids);
        AppendId(kImStartTokenId, &ids);
        AppendEncoded("assistant\n", &ids);
        return ids;
    }

    void LoadVocab(const std::string& vocab_path) {
        std::ifstream ifs(vocab_path);
        if (!ifs) {
            throw std::runtime_error("failed to open vocab: " + vocab_path);
        }
        std::ostringstream oss;
        oss << ifs.rdbuf();
        const std::string text = oss.str();

        std::size_t offset = 0;
        while (offset < text.size()) {
            if (text[offset] != '"') {
                ++offset;
                continue;
            }
            std::string token = DecodeJsonString(text, &offset);
            while (offset < text.size() && std::isspace(static_cast<unsigned char>(text[offset]))) {
                ++offset;
            }
            if (offset >= text.size() || text[offset] != ':') {
                continue;
            }
            ++offset;
            while (offset < text.size() && std::isspace(static_cast<unsigned char>(text[offset]))) {
                ++offset;
            }
            std::size_t number_end = offset;
            while (number_end < text.size() && std::isdigit(static_cast<unsigned char>(text[number_end]))) {
                ++number_end;
            }
            if (number_end == offset) {
                continue;
            }
            const int64_t id = std::stoll(text.substr(offset, number_end - offset));
            token_to_id[token] = id;
            id_to_token[id] = std::move(token);
            offset = number_end;
        }
        if (id_to_token.empty()) {
            throw std::runtime_error("vocab is empty: " + vocab_path);
        }
    }

    void LoadMerges(const std::string& merges_path) {
        std::ifstream ifs(merges_path);
        if (!ifs) {
            throw std::runtime_error("failed to open merges: " + merges_path);
        }
        std::string line;
        int rank = 0;
        while (std::getline(ifs, line)) {
            line = Trim(line);
            if (line.empty() || line[0] == '#') {
                continue;
            }
            const auto sep = line.find(' ');
            if (sep == std::string::npos) {
                continue;
            }
            const std::string first = line.substr(0, sep);
            const std::string second = line.substr(sep + 1);
            bpe_ranks[PairKey(first, second)] = rank++;
        }
        if (bpe_ranks.empty()) {
            throw std::runtime_error("merges are empty: " + merges_path);
        }
    }

    std::vector<std::string> PreTokenize(const std::string& text) const {
        std::vector<std::string> pieces;
        std::size_t i = 0;
        while (i < text.size()) {
            const auto c = static_cast<unsigned char>(text[i]);
            if (c == ' ' && i + 1 < text.size()) {
                const auto next = static_cast<unsigned char>(text[i + 1]);
                if (IsAsciiLetter(next) || next >= 0x80) {
                    const std::size_t start = i++;
                    while (i < text.size()) {
                        const auto ci = static_cast<unsigned char>(text[i]);
                        if (IsAsciiLetter(ci) || ci >= 0x80) {
                            ++i;
                            while (i < text.size() && IsUtf8Continuation(static_cast<unsigned char>(text[i]))) {
                                ++i;
                            }
                        } else {
                            break;
                        }
                    }
                    pieces.push_back(text.substr(start, i - start));
                    continue;
                }
            }
            if (IsAsciiLetter(c) || c >= 0x80) {
                const std::size_t start = i;
                while (i < text.size()) {
                    const auto ci = static_cast<unsigned char>(text[i]);
                    if (IsAsciiLetter(ci) || ci >= 0x80) {
                        ++i;
                        while (i < text.size() && IsUtf8Continuation(static_cast<unsigned char>(text[i]))) {
                            ++i;
                        }
                    } else {
                        break;
                    }
                }
                pieces.push_back(text.substr(start, i - start));
                continue;
            }
            if (IsAsciiDigit(c)) {
                pieces.push_back(text.substr(i, 1));
                ++i;
                continue;
            }
            if (IsWhitespace(c)) {
                const std::size_t start = i;
                while (i < text.size() && IsWhitespace(static_cast<unsigned char>(text[i]))) {
                    ++i;
                }
                pieces.push_back(text.substr(start, i - start));
                continue;
            }
            const std::size_t start = i;
            while (i < text.size()) {
                const auto ci = static_cast<unsigned char>(text[i]);
                if (IsWhitespace(ci) || IsAsciiLetter(ci) || IsAsciiDigit(ci) || ci >= 0x80) {
                    break;
                }
                ++i;
            }
            pieces.push_back(text.substr(start, i - start));
        }
        return pieces;
    }

    std::vector<std::string> ByteEncode(const std::string& piece) const {
        std::vector<std::string> symbols;
        symbols.reserve(piece.size());
        for (unsigned char byte : piece) {
            auto it = byte_encoder.find(byte);
            if (it == byte_encoder.end()) {
                throw std::runtime_error("missing byte encoder entry");
            }
            symbols.push_back(it->second);
        }
        return symbols;
    }

    std::vector<std::string> ApplyBpe(std::vector<std::string> symbols) const {
        if (symbols.size() < 2) {
            return symbols;
        }
        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            std::size_t best_index = symbols.size();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto it = bpe_ranks.find(PairKey(symbols[i], symbols[i + 1]));
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index == symbols.size()) {
                break;
            }
            std::vector<std::string> merged;
            merged.reserve(symbols.size() - 1);
            for (std::size_t i = 0; i < symbols.size(); ++i) {
                if (i == best_index) {
                    merged.push_back(symbols[i] + symbols[i + 1]);
                    ++i;
                } else {
                    merged.push_back(symbols[i]);
                }
            }
            symbols = std::move(merged);
            if (symbols.size() < 2) {
                break;
            }
        }
        return symbols;
    }

    void AppendEncoded(const std::string& text, std::vector<int64_t>* ids) const {
        for (const std::string& piece : PreTokenize(text)) {
            for (const std::string& token : ApplyBpe(ByteEncode(piece))) {
                auto it = token_to_id.find(token);
                if (it == token_to_id.end()) {
                    throw std::runtime_error("token not found in vocab: " + token);
                }
                ids->push_back(it->second);
            }
        }
    }

    static void AppendId(int64_t id, std::vector<int64_t>* ids) {
        ids->push_back(id);
    }

    std::unordered_map<std::string, int64_t> token_to_id;
    std::unordered_map<int64_t, std::string> id_to_token;
    std::unordered_map<unsigned char, std::string> byte_encoder;
    std::unordered_map<uint32_t, unsigned char> byte_decoder;
    std::unordered_map<std::string, int> bpe_ranks;
};

QwenTokenizer::QwenTokenizer(const std::string& vocab_path, const std::string& merges_path)
    : impl_(std::make_unique<Impl>(vocab_path, merges_path)) {}

QwenTokenizer::~QwenTokenizer() = default;
QwenTokenizer::QwenTokenizer(QwenTokenizer&&) noexcept = default;
QwenTokenizer& QwenTokenizer::operator=(QwenTokenizer&&) noexcept = default;

std::vector<int64_t> QwenTokenizer::EncodeChat(
    const std::string& system,
    const std::string& prompt) const {
    return impl_->EncodeChat(system, prompt);
}

std::string QwenTokenizer::Decode(int64_t token_id) const {
    return impl_->Decode(token_id);
}

}  // namespace mini_infer

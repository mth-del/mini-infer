#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mini_infer {

class QwenTokenizer {
public:
    QwenTokenizer(const std::string& vocab_path, const std::string& merges_path);
    ~QwenTokenizer();

    QwenTokenizer(const QwenTokenizer&) = delete;
    QwenTokenizer& operator=(const QwenTokenizer&) = delete;
    QwenTokenizer(QwenTokenizer&&) noexcept;
    QwenTokenizer& operator=(QwenTokenizer&&) noexcept;

    std::vector<int64_t> EncodeChat(const std::string& system, const std::string& prompt) const;
    std::string Decode(int64_t token_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mini_infer

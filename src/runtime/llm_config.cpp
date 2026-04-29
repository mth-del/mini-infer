#include "runtime/llm_config.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mini_infer {
namespace {

std::string ReadTextFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("failed to open config file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::size_t FindJsonValue(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON key: " + key);
    }
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("missing ':' after JSON key: " + key);
    }
    std::size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    if (pos >= text.size()) {
        throw std::runtime_error("missing JSON value for key: " + key);
    }
    return pos;
}

int64_t ReadJsonInt(const std::string& text, const std::string& key) {
    std::size_t pos = FindJsonValue(text, key);
    bool neg = false;
    if (text[pos] == '-') {
        neg = true;
        ++pos;
    }
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        throw std::runtime_error("JSON key is not an integer: " + key);
    }
    int64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = value * 10 + static_cast<int64_t>(text[pos] - '0');
        ++pos;
    }
    return neg ? -value : value;
}

std::vector<int64_t> ReadJsonIntOrArray(const std::string& text, const std::string& key) {
    std::size_t pos = FindJsonValue(text, key);
    std::vector<int64_t> values;
    if (text[pos] != '[') {
        values.push_back(ReadJsonInt(text, key));
        return values;
    }

    ++pos;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos < text.size() && text[pos] == ']') {
            return values;
        }
        bool neg = false;
        if (pos < text.size() && text[pos] == '-') {
            neg = true;
            ++pos;
        }
        if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
            throw std::runtime_error("JSON array contains non-integer for key: " + key);
        }
        int64_t value = 0;
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
            value = value * 10 + static_cast<int64_t>(text[pos] - '0');
            ++pos;
        }
        values.push_back(neg ? -value : value);
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            return values;
        }
    }
    throw std::runtime_error("unterminated integer array for key: " + key);
}

bool ContainsId(const std::vector<int64_t>& values, int64_t id) {
    for (int64_t value : values) {
        if (value == id) {
            return true;
        }
    }
    return false;
}

}  // namespace

LlmModelConfig LoadLlmModelConfig(
    const std::string& config_path,
    const std::string& generation_config_path) {
    const std::string config_json = ReadTextFile(config_path);
    const std::string generation_json = ReadTextFile(generation_config_path);

    LlmModelConfig config;
    config.num_layers = static_cast<int>(ReadJsonInt(config_json, "num_hidden_layers"));
    config.num_kv_heads = static_cast<int>(ReadJsonInt(config_json, "num_key_value_heads"));
    const int hidden_size = static_cast<int>(ReadJsonInt(config_json, "hidden_size"));
    const int num_attention_heads = static_cast<int>(ReadJsonInt(config_json, "num_attention_heads"));
    if (num_attention_heads <= 0 || hidden_size % num_attention_heads != 0) {
        throw std::runtime_error("invalid hidden_size / num_attention_heads in config");
    }
    config.head_dim = hidden_size / num_attention_heads;
    config.vocab_size = static_cast<int>(ReadJsonInt(config_json, "vocab_size"));

    config.bos_token_id = ReadJsonInt(config_json, "bos_token_id");
    config.eos_token_id = ReadJsonInt(config_json, "eos_token_id");
    config.pad_token_id = ReadJsonInt(generation_json, "pad_token_id");
    config.stop_token_ids = ReadJsonIntOrArray(generation_json, "eos_token_id");
    if (!ContainsId(config.stop_token_ids, config.eos_token_id)) {
        config.stop_token_ids.push_back(config.eos_token_id);
    }
    if (!ContainsId(config.stop_token_ids, config.bos_token_id)) {
        config.stop_token_ids.push_back(config.bos_token_id);
    }

    // Qwen2 chat-template special token ids are tokenizer-level constants.
    config.im_start_token_id = 151644;
    config.end_token_id = config.eos_token_id;
    return config;
}

std::string FormatLayerName(const std::string& pattern, int layer) {
    const std::string placeholder = "{layer}";
    std::string out = pattern;
    const std::size_t pos = out.find(placeholder);
    if (pos != std::string::npos) {
        out.replace(pos, placeholder.size(), std::to_string(layer));
    }
    return out;
}

}  // namespace mini_infer

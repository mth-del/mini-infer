#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_infer {

struct LlmModelConfig {
    int num_layers{0};
    int num_kv_heads{0};
    int head_dim{0};
    int vocab_size{0};

    int64_t bos_token_id{-1};
    int64_t eos_token_id{-1};
    int64_t pad_token_id{-1};
    int64_t end_token_id{-1};
    int64_t im_start_token_id{-1};
    std::vector<int64_t> stop_token_ids;

    std::string past_key_pattern{"past_key_values.{layer}.key"};
    std::string past_value_pattern{"past_key_values.{layer}.value"};
    std::string present_key_pattern{"present.{layer}.key"};
    std::string present_value_pattern{"present.{layer}.value"};
};

LlmModelConfig LoadLlmModelConfig(
    const std::string& config_path,
    const std::string& generation_config_path);

std::string FormatLayerName(const std::string& pattern, int layer);

}  // namespace mini_infer

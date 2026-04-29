#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/llm_config.h"
#include "runtime/qwen_tokenizer.h"
#include "runtime/runtime.h"
#ifdef MINI_INFER_HAS_TENSORRT
#include "runtime/tensorrt_backend.h"
#endif
#include "../src/backends/onnx/ort_backend.h"

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

mini_infer::Tensor MakeInt64Tensor(
    std::string name,
    std::vector<int64_t> shape,
    std::vector<int64_t> values) {
    mini_infer::Tensor tensor;
    tensor.name = std::move(name);
    tensor.shape = std::move(shape);
    tensor.elem_type = mini_infer::TensorElementType::INT64;
    tensor.int64_data = std::move(values);
    return tensor;
}

mini_infer::Tensor MakeFloatTensor(std::string name, std::vector<int64_t> shape) {
    mini_infer::Tensor tensor;
    tensor.name = std::move(name);
    tensor.shape = std::move(shape);
    tensor.elem_type = mini_infer::TensorElementType::FLOAT32;
    tensor.data.resize(tensor.numel(), 0.0F);
    return tensor;
}

std::vector<int64_t> Ones(std::size_t n) {
    return std::vector<int64_t>(n, 1);
}

std::vector<int64_t> Positions(std::size_t begin, std::size_t n) {
    std::vector<int64_t> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        values.push_back(static_cast<int64_t>(begin + i));
    }
    return values;
}

std::vector<mini_infer::Tensor> BuildInputs(
    const mini_infer::LlmModelConfig& config,
    const std::vector<int64_t>& all_token_ids,
    const std::vector<int64_t>& current_token_ids,
    const std::unordered_map<std::string, mini_infer::Tensor>& past) {
    const std::size_t past_len = all_token_ids.size() - current_token_ids.size();

    std::vector<mini_infer::Tensor> inputs;
    inputs.reserve(3 + static_cast<std::size_t>(config.num_layers) * 2);
    inputs.push_back(MakeInt64Tensor(
        "input_ids",
        {1, static_cast<int64_t>(current_token_ids.size())},
        current_token_ids));
    inputs.push_back(MakeInt64Tensor(
        "attention_mask",
        {1, static_cast<int64_t>(all_token_ids.size())},
        Ones(all_token_ids.size())));
    inputs.push_back(MakeInt64Tensor(
        "position_ids",
        {1, static_cast<int64_t>(current_token_ids.size())},
        Positions(past_len, current_token_ids.size())));

    for (int layer = 0; layer < config.num_layers; ++layer) {
        for (const std::string& name : {
                 mini_infer::FormatLayerName(config.past_key_pattern, layer),
                 mini_infer::FormatLayerName(config.past_value_pattern, layer)}) {
            auto it = past.find(name);
            if (it != past.end()) {
                inputs.push_back(it->second);
            } else {
                inputs.push_back(MakeFloatTensor(
                    name,
                    {1,
                     config.num_kv_heads,
                     static_cast<int64_t>(past_len),
                     config.head_dim}));
            }
        }
    }
    return inputs;
}

const mini_infer::Tensor& FindOutput(
    const std::vector<mini_infer::Tensor>& outputs,
    const std::string& name) {
    auto it = std::find_if(outputs.begin(), outputs.end(), [&](const mini_infer::Tensor& tensor) {
        return tensor.name == name;
    });
    if (it == outputs.end()) {
        throw std::runtime_error("missing output tensor: " + name);
    }
    return *it;
}

int64_t ArgmaxLastToken(const mini_infer::Tensor& logits, int expected_vocab_size) {
    if (logits.elem_type != mini_infer::TensorElementType::FLOAT32 || logits.shape.size() != 3) {
        throw std::runtime_error("logits must be float32 [batch, seq, vocab]");
    }
    const int64_t seq_len = logits.shape[1];
    const int64_t vocab_size = logits.shape[2];
    if (seq_len <= 0 || vocab_size <= 0 || vocab_size != expected_vocab_size) {
        throw std::runtime_error("unexpected logits shape");
    }
    const std::size_t offset = static_cast<std::size_t>((seq_len - 1) * vocab_size);
    const auto begin = logits.data.begin() + static_cast<std::ptrdiff_t>(offset);
    const auto end = begin + static_cast<std::ptrdiff_t>(vocab_size);
    return static_cast<int64_t>(std::distance(begin, std::max_element(begin, end)));
}

std::unordered_map<std::string, mini_infer::Tensor> ExtractPast(
    const mini_infer::LlmModelConfig& config,
    const std::vector<mini_infer::Tensor>& outputs) {
    std::unordered_map<std::string, mini_infer::Tensor> past;
    past.reserve(static_cast<std::size_t>(config.num_layers) * 2);
    for (int layer = 0; layer < config.num_layers; ++layer) {
        mini_infer::Tensor key = FindOutput(
            outputs, mini_infer::FormatLayerName(config.present_key_pattern, layer));
        key.name = mini_infer::FormatLayerName(config.past_key_pattern, layer);
        past[key.name] = std::move(key);

        mini_infer::Tensor value = FindOutput(
            outputs, mini_infer::FormatLayerName(config.present_value_pattern, layer));
        value.name = mini_infer::FormatLayerName(config.past_value_pattern, layer);
        past[value.name] = std::move(value);
    }
    return past;
}

void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--model <model.onnx>] [--vocab <vocab.json>] [--merges <merges.txt>]"
              << " [--config <config.json>] [--generation-config <generation_config.json>]"
              << " [--prompt <text>] [--system <text>]"
              << " [--provider cpu|cuda|tensorrt] [--max-new-tokens N]\n";
}

std::string Trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string GpuMemorySnapshot() {
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (!pipe) {
        return "unavailable";
    }

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    const int status = pclose(pipe);
    if (status != 0 || output.empty()) {
        return "unavailable";
    }

    std::istringstream lines(output);
    std::string line;
    std::ostringstream result;
    int gpu_id = 0;
    bool wrote = false;
    while (std::getline(lines, line)) {
        line = Trim(line);
        if (line.empty()) {
            continue;
        }
        const auto comma = line.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::string used = Trim(line.substr(0, comma));
        const std::string total = Trim(line.substr(comma + 1));
        if (wrote) {
            result << ' ';
        }
        result << "gpu" << gpu_id << "_used_mib=" << used
               << " gpu" << gpu_id << "_total_mib=" << total;
        wrote = true;
        ++gpu_id;
    }
    return wrote ? result.str() : "unavailable";
}

bool IsStopToken(const mini_infer::LlmModelConfig& config, int64_t token_id) {
    for (int64_t stop_id : config.stop_token_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_q4.onnx";
    std::string vocab_path = "models/Qwen2.5-1.5B-Instruct-ONNX/vocab.json";
    std::string merges_path = "models/Qwen2.5-1.5B-Instruct-ONNX/merges.txt";
    std::string config_path = "models/Qwen2.5-1.5B-Instruct-ONNX/config.json";
    std::string generation_config_path =
        "models/Qwen2.5-1.5B-Instruct-ONNX/generation_config.json";
    std::string prompt = "你好";
    std::string system = "You are a helpful assistant.";
    std::string provider = "cuda";
    int max_new_tokens = 2;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--generation-config" && i + 1 < argc) {
            generation_config_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            system = argv[++i];
        } else if (arg == "--provider" && i + 1 < argc) {
            provider = argv[++i];
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }

    if (provider != "cpu" && provider != "cuda" && provider != "tensorrt") {
        std::cerr << "--provider must be cpu, cuda, or tensorrt\n";
        return 2;
    }
#ifndef MINI_INFER_HAS_TENSORRT
    if (provider == "tensorrt") {
        std::cerr << "TensorRT backend was not enabled in this build\n";
        return 2;
    }
#endif
    if (max_new_tokens < 1) {
        max_new_tokens = 1;
    }

    try {
        const auto total_start = Clock::now();
        const std::string gpu_before_session = GpuMemorySnapshot();
        const auto session_start = Clock::now();
        mini_infer::LlmModelConfig model_config =
            mini_infer::LoadLlmModelConfig(config_path, generation_config_path);
        mini_infer::QwenTokenizer tokenizer(vocab_path, merges_path);
        std::vector<int64_t> token_ids = tokenizer.EncodeChat(system, prompt);
        std::vector<int64_t> current_ids = token_ids;
        mini_infer::Runtime runtime;
        std::shared_ptr<mini_infer::Backend> backend;
#ifdef MINI_INFER_HAS_TENSORRT
        if (provider == "tensorrt") {
            backend = std::make_shared<mini_infer::TensorRtBackend>(model_path, 0);
        } else
#endif
        {
            backend = std::make_shared<mini_infer::OrtBackend>(
                model_path, provider == "cuda", 0);
        }
        runtime.set_backend(backend);
        if (!backend->init()) {
            std::cerr << "Backend init failed\n";
            return 1;
        }
        if (!runtime.load_model(model_path)) {
            std::cerr << "Failed to load model path\n";
            return 1;
        }
        const auto session_end = Clock::now();
        const std::string gpu_after_session = GpuMemorySnapshot();

        std::cout << "[prompt_tokens]";
        for (int64_t id : token_ids) {
            std::cout << " " << id;
        }
        std::cout << "\n";

        std::unordered_map<std::string, mini_infer::Tensor> past;
        std::vector<int64_t> generated;
        std::string generated_text;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        std::string gpu_after_prefill = "not_run";

        for (int step = 0; step < max_new_tokens; ++step) {
            auto inputs = BuildInputs(model_config, token_ids, current_ids, past);
            const auto infer_start = Clock::now();
            auto outputs = runtime.infer_many(inputs);
            const auto infer_end = Clock::now();
            const double infer_ms = MsSince(infer_start, infer_end);
            if (step == 0) {
                prefill_ms += infer_ms;
                gpu_after_prefill = GpuMemorySnapshot();
            } else {
                decode_ms += infer_ms;
            }

            const auto& logits = FindOutput(outputs, "logits");
            const int64_t next_id = ArgmaxLastToken(logits, model_config.vocab_size);
            std::cout << "[token] step=" << step
                      << " id=" << next_id
                      << " infer_ms="
                      << std::fixed << std::setprecision(3)
                      << infer_ms
                      << "\n";

            if (IsStopToken(model_config, next_id)) {
                std::cout << "[stop] eos_token_id=" << next_id << "\n";
                break;
            }

            generated.push_back(next_id);
            const std::string piece = tokenizer.Decode(next_id);
            generated_text += piece;
            std::cout << "[piece] " << piece << "\n";

            token_ids.push_back(next_id);
            current_ids = {next_id};
            past = ExtractPast(model_config, outputs);
        }

        const auto total_end = Clock::now();
        const std::string gpu_after_decode = GpuMemorySnapshot();
        std::cout << "[generated_ids]";
        for (int64_t id : generated) {
            std::cout << " " << id;
        }
        std::cout << "\n";
        std::cout << "[assistant] " << generated_text << "\n";
        const double total_gen_ms = prefill_ms + decode_ms;
        const double decode_avg_ms = decode_ms / static_cast<double>(std::max<int>(generated.size() - 1, 1));
        const double tokens_per_second = total_gen_ms > 0.0
            ? static_cast<double>(generated.size()) / (total_gen_ms / 1000.0)
            : 0.0;
        std::cout << "[timing] "
                  << std::fixed << std::setprecision(3)
                  << "session_init_ms=" << MsSince(session_start, session_end) << " "
                  << "prefill_ms=" << prefill_ms << " "
                  << "decode_ms=" << decode_ms << " "
                  << "decode_avg_ms=" << decode_avg_ms << " "
                  << "generated_tokens=" << generated.size() << " "
                  << "tokens_per_second=" << tokens_per_second << " "
                  << "total_ms=" << MsSince(total_start, total_end)
                  << "\n";
        std::cout << "[gpu_memory] "
                  << "before_session=(" << gpu_before_session << ") "
                  << "after_session=(" << gpu_after_session << ") "
                  << "after_prefill=(" << gpu_after_prefill << ") "
                  << "after_decode=(" << gpu_after_decode << ")"
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "qwen_cli_cpp failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << "\n";
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const {
        delete ptr;
    }
};

enum class PrecisionMode {
    FP32,
    FP16,
    FP16_FP32_SENSITIVE
};

struct ShapeSpec {
    std::string name;
    nvinfer1::Dims min_dims{};
    nvinfer1::Dims opt_dims{};
    nvinfer1::Dims max_dims{};
};

struct Options {
    std::string onnx_path;
    std::string engine_path;
    PrecisionMode precision{PrecisionMode::FP16};
    std::size_t workspace_mib{4096};
    std::vector<ShapeSpec> shapes;
    std::vector<std::string> fp32_patterns;
    std::vector<std::string> fp32_types;
    bool qwen_profile{false};
    int layers{28};
    int kv_heads{2};
    int head_dim{128};
    int seq_len{20};
    int past_len{0};
    int seq_len_min{-1};
    int seq_len_opt{-1};
    int seq_len_max{-1};
    int past_len_min{-1};
    int past_len_opt{-1};
    int past_len_max{-1};
};

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool ContainsAny(const std::string& text, const std::vector<std::string>& needles) {
    for (const std::string& needle : needles) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims ParseDims(const std::string& text) {
    nvinfer1::Dims dims{};
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find('x', begin);
        const std::string token = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (token.empty()) {
            throw std::runtime_error("empty dimension in shape: " + text);
        }
        if (dims.nbDims >= nvinfer1::Dims::MAX_DIMS) {
            throw std::runtime_error("shape rank exceeds TensorRT max dims: " + text);
        }
        dims.d[dims.nbDims++] = std::stoi(token);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return dims;
}

ShapeSpec ParseShapeSpec(const std::string& spec) {
    const std::size_t equal = spec.find('=');
    if (equal == std::string::npos || equal == 0 || equal + 1 >= spec.size()) {
        throw std::runtime_error("shape must be name=dimxdim..., got: " + spec);
    }
    const nvinfer1::Dims dims = ParseDims(spec.substr(equal + 1));
    return {spec.substr(0, equal), dims, dims, dims};
}

void AddQwenProfileShapes(Options& options) {
    auto dims2 = [](int second) {
        return ParseDims("1x" + std::to_string(second));
    };
    auto kv_dims = [&](int past_len) {
        return ParseDims(
            "1x" + std::to_string(options.kv_heads) + "x" +
            std::to_string(past_len) + "x" + std::to_string(options.head_dim));
    };

    options.shapes.push_back({
        "input_ids",
        dims2(options.seq_len_min),
        dims2(options.seq_len_opt),
        dims2(options.seq_len_max)});
    options.shapes.push_back({
        "attention_mask",
        dims2(options.past_len_min + options.seq_len_min),
        dims2(options.past_len_opt + options.seq_len_opt),
        dims2(options.past_len_max + options.seq_len_max)});
    options.shapes.push_back({
        "position_ids",
        dims2(options.seq_len_min),
        dims2(options.seq_len_opt),
        dims2(options.seq_len_max)});
    for (int layer = 0; layer < options.layers; ++layer) {
        const std::string prefix = "past_key_values." + std::to_string(layer);
        options.shapes.push_back({
            prefix + ".key",
            kv_dims(options.past_len_min),
            kv_dims(options.past_len_opt),
            kv_dims(options.past_len_max)});
        options.shapes.push_back({
            prefix + ".value",
            kv_dims(options.past_len_min),
            kv_dims(options.past_len_opt),
            kv_dims(options.past_len_max)});
    }
}

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --onnx model.onnx --engine model.engine [options]\n"
        << "Options:\n"
        << "  --precision fp32|fp16|fp16_fp32_sensitive\n"
        << "  --shape name=dimxdim...              Add a fixed-shape optimization profile input\n"
        << "  --force-fp32-pattern text            Force layers whose names contain text to FP32\n"
        << "  --force-fp32-type type               Force TensorRT layer type to FP32\n"
        << "  --qwen-profile                       Add Qwen KV-cache input shapes automatically\n"
        << "  --seq-len N --past-len N             Qwen current/past sequence lengths\n"
        << "  --seq-len-min/opt/max N              Qwen current length profile range\n"
        << "  --past-len-min/opt/max N             Qwen KV-cache length profile range\n"
        << "  --layers N --kv-heads N --head-dim N Qwen KV-cache layout\n"
        << "  --workspace-mib N                    TensorRT workspace limit, default 4096\n"
        << "Types: matrix_multiply, softmax, ragged_softmax, elementwise, unary, reduce, shuffle, scale\n";
}

PrecisionMode ParsePrecision(const std::string& text) {
    if (text == "fp32") {
        return PrecisionMode::FP32;
    }
    if (text == "fp16") {
        return PrecisionMode::FP16;
    }
    if (text == "fp16_fp32_sensitive") {
        return PrecisionMode::FP16_FP32_SENSITIVE;
    }
    throw std::runtime_error("unknown precision mode: " + text);
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--onnx") {
            options.onnx_path = need_value("--onnx");
        } else if (arg == "--engine") {
            options.engine_path = need_value("--engine");
        } else if (arg == "--precision") {
            options.precision = ParsePrecision(need_value("--precision"));
        } else if (arg == "--shape") {
            options.shapes.push_back(ParseShapeSpec(need_value("--shape")));
        } else if (arg == "--force-fp32-pattern") {
            options.fp32_patterns.push_back(ToLower(need_value("--force-fp32-pattern")));
        } else if (arg == "--force-fp32-type") {
            options.fp32_types.push_back(ToLower(need_value("--force-fp32-type")));
        } else if (arg == "--qwen-profile") {
            options.qwen_profile = true;
        } else if (arg == "--seq-len") {
            options.seq_len = std::stoi(need_value("--seq-len"));
        } else if (arg == "--past-len") {
            options.past_len = std::stoi(need_value("--past-len"));
        } else if (arg == "--seq-len-min") {
            options.seq_len_min = std::stoi(need_value("--seq-len-min"));
        } else if (arg == "--seq-len-opt") {
            options.seq_len_opt = std::stoi(need_value("--seq-len-opt"));
        } else if (arg == "--seq-len-max") {
            options.seq_len_max = std::stoi(need_value("--seq-len-max"));
        } else if (arg == "--past-len-min") {
            options.past_len_min = std::stoi(need_value("--past-len-min"));
        } else if (arg == "--past-len-opt") {
            options.past_len_opt = std::stoi(need_value("--past-len-opt"));
        } else if (arg == "--past-len-max") {
            options.past_len_max = std::stoi(need_value("--past-len-max"));
        } else if (arg == "--layers") {
            options.layers = std::stoi(need_value("--layers"));
        } else if (arg == "--kv-heads") {
            options.kv_heads = std::stoi(need_value("--kv-heads"));
        } else if (arg == "--head-dim") {
            options.head_dim = std::stoi(need_value("--head-dim"));
        } else if (arg == "--workspace-mib") {
            options.workspace_mib = static_cast<std::size_t>(std::stoull(need_value("--workspace-mib")));
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + arg);
        }
    }

    if (options.onnx_path.empty() || options.engine_path.empty()) {
        throw std::runtime_error("--onnx and --engine are required");
    }
    if (options.seq_len_min < 0) options.seq_len_min = options.seq_len;
    if (options.seq_len_opt < 0) options.seq_len_opt = options.seq_len;
    if (options.seq_len_max < 0) options.seq_len_max = options.seq_len;
    if (options.past_len_min < 0) options.past_len_min = options.past_len;
    if (options.past_len_opt < 0) options.past_len_opt = options.past_len;
    if (options.past_len_max < 0) options.past_len_max = options.past_len;

    if (options.seq_len_min < 1 || options.seq_len_opt < options.seq_len_min ||
        options.seq_len_max < options.seq_len_opt || options.past_len_min < 0 ||
        options.past_len_opt < options.past_len_min ||
        options.past_len_max < options.past_len_opt || options.layers < 1 ||
        options.kv_heads < 1 || options.head_dim < 1) {
        throw std::runtime_error("invalid Qwen profile dimensions");
    }
    if (options.qwen_profile) {
        AddQwenProfileShapes(options);
    }
    if (options.shapes.empty()) {
        throw std::runtime_error("at least one --shape or --qwen-profile is required");
    }
    return options;
}

bool IsSensitiveLayer(nvinfer1::ILayer& layer) {
    const std::string name = ToLower(layer.getName() ? layer.getName() : "");
    if (ContainsAny(name, {"layernorm", "layer_norm", "rmsnorm", "rms_norm", "reduce", "pow"})) {
        return true;
    }
    if (layer.getType() == nvinfer1::LayerType::kREDUCE) {
        return true;
    }
    if (layer.getType() == nvinfer1::LayerType::kUNARY) {
        auto* unary = static_cast<nvinfer1::IUnaryLayer*>(&layer);
        return unary->getOperation() == nvinfer1::UnaryOperation::kSQRT;
    }
    if (layer.getType() == nvinfer1::LayerType::kELEMENTWISE) {
        auto* elementwise = static_cast<nvinfer1::IElementWiseLayer*>(&layer);
        return elementwise->getOperation() == nvinfer1::ElementWiseOperation::kPOW;
    }
    return false;
}

bool MatchesLayerType(nvinfer1::LayerType type, const std::string& name) {
    if (name == "matrix_multiply") {
        return type == nvinfer1::LayerType::kMATRIX_MULTIPLY;
    }
    if (name == "softmax") {
        return type == nvinfer1::LayerType::kSOFTMAX;
    }
    if (name == "ragged_softmax") {
        return type == nvinfer1::LayerType::kRAGGED_SOFTMAX;
    }
    if (name == "elementwise") {
        return type == nvinfer1::LayerType::kELEMENTWISE;
    }
    if (name == "unary") {
        return type == nvinfer1::LayerType::kUNARY;
    }
    if (name == "reduce") {
        return type == nvinfer1::LayerType::kREDUCE;
    }
    if (name == "shuffle") {
        return type == nvinfer1::LayerType::kSHUFFLE;
    }
    if (name == "scale") {
        return type == nvinfer1::LayerType::kSCALE;
    }
    throw std::runtime_error("unsupported --force-fp32-type: " + name);
}

bool MatchesCustomFp32Rule(nvinfer1::ILayer& layer, const Options& options) {
    const std::string layer_name = ToLower(layer.getName() ? layer.getName() : "");
    if (ContainsAny(layer_name, options.fp32_patterns)) {
        return true;
    }
    for (const std::string& type : options.fp32_types) {
        if (MatchesLayerType(layer.getType(), type)) {
            return true;
        }
    }
    return false;
}

bool HasFloatingOutput(nvinfer1::ILayer& layer) {
    for (int32_t output = 0; output < layer.getNbOutputs(); ++output) {
        nvinfer1::ITensor* tensor = layer.getOutput(output);
        if (!tensor) {
            continue;
        }
        const nvinfer1::DataType type = tensor->getType();
        if (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) {
            return true;
        }
    }
    return false;
}

bool CanSetFp32Precision(nvinfer1::ILayer& layer) {
    const nvinfer1::LayerType type = layer.getType();
    if (type == nvinfer1::LayerType::kCONSTANT ||
        type == nvinfer1::LayerType::kSHAPE ||
        type == nvinfer1::LayerType::kGATHER) {
        return false;
    }
    return HasFloatingOutput(layer);
}

int ForceLayersToFp32(nvinfer1::INetworkDefinition& network, const Options& options) {
    int count = 0;
    int skipped = 0;
    for (int32_t i = 0; i < network.getNbLayers(); ++i) {
        nvinfer1::ILayer* layer = network.getLayer(i);
        if (!layer) {
            continue;
        }

        const bool force_sensitive =
            options.precision == PrecisionMode::FP16_FP32_SENSITIVE && IsSensitiveLayer(*layer);
        if (!force_sensitive && !MatchesCustomFp32Rule(*layer, options)) {
            continue;
        }
        if (!CanSetFp32Precision(*layer)) {
            ++skipped;
            continue;
        }
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        for (int32_t output = 0; output < layer->getNbOutputs(); ++output) {
            nvinfer1::ITensor* tensor = layer->getOutput(output);
            if (tensor &&
                (tensor->getType() == nvinfer1::DataType::kFLOAT ||
                 tensor->getType() == nvinfer1::DataType::kHALF)) {
                layer->setOutputType(output, nvinfer1::DataType::kFLOAT);
            }
        }
        std::cout << "[precision] force_fp32 layer="
                  << (layer->getName() ? layer->getName() : "<unnamed>") << "\n";
        ++count;
    }
    if (skipped > 0) {
        std::cout << "[precision] skipped_non_float_layers=" << skipped << "\n";
    }
    return count;
}

void WriteBinaryFile(const std::string& path, const void* data, std::size_t size) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("failed to open engine output: " + path);
    }
    ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!ofs) {
        throw std::runtime_error("failed to write engine output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = ParseArgs(argc, argv);
        TrtLogger logger;

        std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>> builder(
            nvinfer1::createInferBuilder(logger));
        if (!builder) {
            throw std::runtime_error("createInferBuilder failed");
        }

        const uint32_t flags =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>> network(
            builder->createNetworkV2(flags));
        if (!network) {
            throw std::runtime_error("createNetworkV2 failed");
        }

        std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>> parser(
            nvonnxparser::createParser(*network, logger));
        if (!parser) {
            throw std::runtime_error("createParser failed");
        }
        if (!parser->parseFromFile(
                options.onnx_path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                const nvonnxparser::IParserError* error = parser->getError(i);
                if (error) {
                    std::cerr << "[parser] " << error->desc() << "\n";
                }
            }
            throw std::runtime_error("parseFromFile failed");
        }

        std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>> config(
            builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("createBuilderConfig failed");
        }
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            options.workspace_mib * 1024ULL * 1024ULL);

        if (options.precision != PrecisionMode::FP32) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        if (options.precision == PrecisionMode::FP16_FP32_SENSITIVE ||
            !options.fp32_patterns.empty() ||
            !options.fp32_types.empty()) {
            const int forced = ForceLayersToFp32(*network, options);
            config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
            std::cout << "[precision] forced_fp32_layers=" << forced << "\n";
        }

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (!profile) {
            throw std::runtime_error("createOptimizationProfile failed");
        }
        for (const ShapeSpec& shape : options.shapes) {
            if (!profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kMIN, shape.min_dims) ||
                !profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kOPT, shape.opt_dims) ||
                !profile->setDimensions(shape.name.c_str(), nvinfer1::OptProfileSelector::kMAX, shape.max_dims)) {
                throw std::runtime_error("setDimensions failed for input: " + shape.name);
            }
        }
        if (!profile->isValid()) {
            throw std::runtime_error("optimization profile is invalid");
        }
        config->addOptimizationProfile(profile);

        std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            throw std::runtime_error("buildSerializedNetwork failed");
        }
        WriteBinaryFile(options.engine_path, serialized->data(), serialized->size());
        std::cout << "[engine] wrote " << options.engine_path
                  << " bytes=" << serialized->size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "trt_build_engine failed: " << e.what() << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
}

#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "runtime/backend.h"
#include "runtime/config.h"
#include "runtime/runtime.h"
#include "runtime/tensorrt_backend.h"
#include "runtime/yolov5_postprocess.h"
#include "../src/backends/onnx/ort_backend.h"
#include "onnxruntime_cxx_api.h"

namespace mini_infer {

namespace {

struct ImageInfo {
    int width{0};
    int height{0};
};

struct ModelInputSpec {
    std::string name;
    std::vector<int64_t> shape;
};

bool ReadPpmHeader(const std::string& image_path, ImageInfo* info) {
    std::ifstream ifs(image_path, std::ios::binary);
    if (!ifs) {
        return false;
    }

    std::string magic;
    ifs >> magic;
    if (magic != "P6") {
        return false;
    }

    ifs >> info->width >> info->height;
    int maxval = 0;
    ifs >> maxval;
    return ifs.good() && info->width > 0 && info->height > 0 && maxval > 0;
}

LetterboxMeta BuildLetterboxMeta(int orig_w, int orig_h, int target_w, int target_h) {
    const float scale = std::min(
        static_cast<float>(target_w) / static_cast<float>(orig_w),
        static_cast<float>(target_h) / static_cast<float>(orig_h));
    const float resized_w = static_cast<float>(orig_w) * scale;
    const float resized_h = static_cast<float>(orig_h) * scale;
    return {
        scale,
        (static_cast<float>(target_w) - resized_w) * 0.5F,
        (static_cast<float>(target_h) - resized_h) * 0.5F,
        orig_w,
        orig_h};
}

// 根据模型输入获取tensor
ModelInputSpec ReadModelInputSpec(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mini_infer_cli_probe");
    Ort::SessionOptions so;
    Ort::Session session(env, model_path.c_str(), so);
    Ort::AllocatorWithDefaultOptions allocator;

    if (session.GetInputCount() == 0) {
        throw std::runtime_error("Model has no inputs");
    }

    auto name = session.GetInputNameAllocated(0, allocator);
    auto type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return {name.get(), tensor_info.GetShape()};
}

Tensor BuildInputLikeModel(const ModelInputSpec& spec) {
    std::vector<int64_t> shape = spec.shape;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            if (i == 0) {
                shape[i] = 1;
            } else if (i == 1) {
                shape[i] = 3;
            } else {
                shape[i] = 640;
            }
        }
    }

    std::size_t numel = 1;
    for (int64_t d : shape) {
        numel *= static_cast<std::size_t>(d);
    }

    Tensor input;
    input.name = spec.name;
    input.shape = std::move(shape);
    input.data.resize(numel, 0.5F);
    return input;
}

std::shared_ptr<Backend> CreateBackend(BackendKind kind, const std::string& model_path, int device_id) {
    switch (kind) {
        case BackendKind::ONNX_CPU:
            return std::make_shared<OrtBackend>(model_path, false, 0);
        case BackendKind::ONNX_CUDA:
            return std::make_shared<OrtBackend>(model_path, true, device_id);
        case BackendKind::TENSORRT:
            return std::make_shared<TensorRtBackend>(model_path, device_id);
        case BackendKind::CPU:
        case BackendKind::OPENVINO:
            break;
    }
    throw std::invalid_argument("Backend not implemented in infer_cli: " + std::string(ToString(kind)));
}

struct BenchResult {
    double avg_ms{0.0};
    std::size_t detections{0};
};

BenchResult RunBenchmark(
    BackendKind kind,
    const std::string& model_path,
    int device_id,
    const Tensor& input,
    const LetterboxMeta& meta,
    int warmup,
    int iters) {
    Runtime runtime;
    auto backend = CreateBackend(kind, model_path, device_id);
    runtime.set_backend(backend);

    if (!backend->init()) {
        throw std::runtime_error("Backend init failed: " + backend->name());
    }
    if (!runtime.load_model(model_path)) {
        throw std::runtime_error("Failed to load model path: " + model_path);
    }

    for (int i = 0; i < warmup; ++i) {
        (void)runtime.infer({input});
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iters));
    Tensor last_out;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_out = runtime.infer({input});
        const auto t1 = std::chrono::steady_clock::now();
        latencies_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    const YoloV5Output yolo = DecodeYoloV5Output(last_out, meta, 0.25F, 0.45F);
    return {sum / static_cast<double>(latencies_ms.size()), yolo.dets.size()};
}

}  // namespace

}  // namespace mini_infer

int main(int argc, char** argv) {
    // 参数:
    // -m <model_path>  模型路径
    // -d <backend>     后端名称(onnx_cpu/onnx_cuda)
    // -i <device_id>   设备ID
    // -l <repeat>      普通推理轮数
    // --bench [warmup] [iters] 基准模式
    std::string model_path = "models/yolov5s.onnx";
    std::string backend_str = "onnx_cpu";
    int device_id = 0;
    bool bench_mode = false;
    int repeat = 1;
    int warmup = 10;
    int iters = 50;

    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "-m" && idx + 1 < argc) {
            model_path = argv[++idx];
        } else if (arg == "-d" && idx + 1 < argc) {
            backend_str = argv[++idx];
        } else if (arg == "-i" && idx + 1 < argc) {
            device_id = std::stoi(argv[++idx]);
        } else if (arg == "-l" && idx + 1 < argc) {
            repeat = std::stoi(argv[++idx]);
            if (repeat < 1) {
                repeat = 1;
            }
        } else if (arg == "--bench") {
            bench_mode = true;
            if (idx + 1 < argc && argv[idx + 1][0] != '-') {
                warmup = std::stoi(argv[++idx]);
            }
            if (idx + 1 < argc && argv[idx + 1][0] != '-') {
                iters = std::stoi(argv[++idx]);
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: infer_cli -m <model> -d <backend> -i <device_id> -l <repeat> [--bench [warmup] [iters]]\n"
                << "  -m model path (default: models/yolov5s.onnx)\n"
                << "  -d backend: onnx_cpu | onnx_cuda | tensorrt (default: onnx_cpu)\n"
                << "  -i device id (default: 0)\n"
                << "  -l inference repeats in normal mode (default: 1)\n"
                << "  --bench [warmup] [iters] benchmark mode (default warmup=10 iters=50)\n";
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return 2;
        }
    }

    mini_infer::BackendKind kind = mini_infer::BackendKind::ONNX_CPU;
    try {
        kind = mini_infer::ParseBackendKind(backend_str);
    } catch (const std::exception& e) {
        std::cerr << "Invalid backend arg: " << backend_str << " (" << e.what() << ")\n";
        return 3;
    }

    mini_infer::ModelInputSpec input_spec;
    try {
        input_spec = mini_infer::ReadModelInputSpec(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to inspect model input: " << e.what() << "\n";
        return 6;
    }
    mini_infer::Tensor input = mini_infer::BuildInputLikeModel(input_spec);
    // Input tensor is generated from model-declared shape and filled with constant values.
    int target_h = 640;
    int target_w = 640;
    if (input.shape.size() >= 4) {
        target_h = static_cast<int>(input.shape[2]);
        target_w = static_cast<int>(input.shape[3]);
    }
    mini_infer::LetterboxMeta meta = mini_infer::BuildLetterboxMeta(
        target_w,
        target_h,
        target_w,
        target_h);

    std::cout << "Model: " << model_path << "\n";
    std::cout << "Input: name=" << input.name << " shape=[";
    for (std::size_t i = 0; i < input.shape.size(); ++i) {
        std::cout << input.shape[i] << (i + 1 < input.shape.size() ? "," : "");
    }
    std::cout << "]\n";

    if (bench_mode) {
        try {
            const auto cpu_res = mini_infer::RunBenchmark(
                mini_infer::BackendKind::ONNX_CPU,
                model_path,
                0,
                input,
                meta,
                warmup,
                iters);
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "[bench] baseline=onnx_cpu avg_ms=" << cpu_res.avg_ms
                      << " dets=" << cpu_res.detections << "\n";

            if (kind != mini_infer::BackendKind::ONNX_CPU) {
                const auto tgt_res = mini_infer::RunBenchmark(
                    kind,
                    model_path,
                    device_id,
                    input,
                    meta,
                    warmup,
                    iters);
                const double speedup = cpu_res.avg_ms / tgt_res.avg_ms;
                std::cout << "[bench] target=" << mini_infer::ToString(kind)
                          << " avg_ms=" << tgt_res.avg_ms
                          << " dets=" << tgt_res.detections
                          << " speedup_vs_cpu=" << speedup << "x\n";
            } else {
                std::cout << "[bench] target is baseline backend (onnx_cpu)\n";
            }
            std::cout << "[bench] warmup=" << warmup << " iters=" << iters << "\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Benchmark failed: " << e.what() << "\n";
            return 4;
        }
    }

    try {
        mini_infer::Runtime runtime;
        // 创建推理后端
        auto backend = mini_infer::CreateBackend(kind, model_path, device_id);
        runtime.set_backend(backend);
        if (!backend->init()) {
            std::cerr << "Backend init failed: " << backend->name() << "\n";
            return 4;
        }
        if (!runtime.load_model(model_path)) {
            std::cerr << "Failed to load model path.\n";
            return 1;
        }

        mini_infer::Tensor output;
        double total_ms = 0.0;
        for (int i = 0; i < repeat; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            output = runtime.infer({input});
            const auto t1 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        const double infer_ms = total_ms / static_cast<double>(repeat);
        const double infer_fps = infer_ms > 0.0 ? (1000.0 / infer_ms) : 0.0;
        mini_infer::YoloV5Output yolo = mini_infer::DecodeYoloV5Output(output, meta, 0.25F, 0.45F);

        std::cout << "Backend: " << backend->name() << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Inference repeats: " << repeat << "\n";
        std::cout << "Inference time(avg): " << infer_ms << " ms"
                  << " (" << infer_fps << " FPS)\n";
        std::cout << "Output tensor shape: [1,25200,85]\n";
        std::cout << "Detections after NMS: " << yolo.dets.size() << "\n";
        for (std::size_t i = 0; i < yolo.dets.size(); ++i) {
            const auto& d = yolo.dets[i];
            std::cout << "#" << i
                      << " cls=" << d.class_id
                      << " score=" << d.score
                      << " box=(" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << ")\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << "\n";
        return 5;
    }

    return 0;
}

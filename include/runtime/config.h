#pragma once

#include <stdexcept>
#include <string>

namespace mini_infer {

enum class BackendKind {
    CPU,
    ONNX_CPU,
    ONNX_CUDA,
    TENSORRT,
    OPENVINO
};

inline BackendKind ParseBackendKind(const std::string& s) {
    if (s == "cpu") return BackendKind::CPU;
    if (s == "onnx_cpu") return BackendKind::ONNX_CPU;
    if (s == "onnx_cuda") return BackendKind::ONNX_CUDA;
    if (s == "tensorrt") return BackendKind::TENSORRT;
    if (s == "openvino") return BackendKind::OPENVINO;
    throw std::invalid_argument("Unknown backend: " + s);
}

inline const char* ToString(BackendKind kind) {
    switch (kind) {
        case BackendKind::CPU: return "cpu";
        case BackendKind::ONNX_CPU: return "onnx_cpu";
        case BackendKind::ONNX_CUDA: return "onnx_cuda";
        case BackendKind::TENSORRT: return "tensorrt";
        case BackendKind::OPENVINO: return "openvino";
    }
    return "unknown";
}

struct RuntimeConfig {
    BackendKind backend{BackendKind::ONNX_CPU};
    int device_id{0};
    float conf_thres{0.25F};
    float nms_thres{0.45F};
};

}  // namespace mini_infer
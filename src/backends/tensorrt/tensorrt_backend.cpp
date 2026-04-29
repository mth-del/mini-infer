#include "runtime/tensorrt_backend.h"

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>

namespace mini_infer {

TensorRtBackend::TensorRtBackend(std::string model_path, int device_id)
    : model_path_(std::move(model_path)), device_id_(device_id) {}

bool TensorRtBackend::init() {
    // Minimal runtime check so CLI can switch backend safely.
    // We don't build a TensorRT engine yet in this step.
    void* handle = dlopen("libnvinfer.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "TensorRT backend unavailable: libnvinfer.so not found.\n";
        std::cerr << "Install TensorRT runtime and ensure library path is visible.\n";
        return false;
    }
    dlclose(handle);
    ready_ = true;
    std::cout << "TensorRT backend initialized (skeleton mode), device_id=" << device_id_
              << ", model=" << model_path_ << "\n";
    return true;
}

Tensor TensorRtBackend::run(const std::vector<Tensor>& inputs) {
    if (!ready_) {
        throw std::runtime_error("TensorRtBackend is not initialized");
    }
    if (inputs.empty()) {
        return {};
    }
    // TODO: Replace with real TensorRT engine execution.
    // For now, keep shape-compatible behavior and make backend selectable.
    return inputs.front();
}

}  // namespace mini_infer

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "runtime/backend.h"

namespace mini_infer {
    class OrtBackend final : public Backend {
    public:
        OrtBackend(std::string model_path, bool use_cuda = false, int device_id = 0);
        ~OrtBackend() override;   // 显式声明
        std::string name() const override { return use_cuda_ ? "onnxruntime-cuda" : "onnxruntime-cpu"; }
        bool init() override;
        Tensor run(const std::vector<Tensor>& inputs) override;
    private:
        std::string model_path_;
        bool use_cuda_{false};
        int device_id_{0};
        // PIMPL: 避免头文件暴露 ORT 细节
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    } 
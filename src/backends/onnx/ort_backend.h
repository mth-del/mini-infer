#pragma once
#include <memory>
#include <string>
#include <vector>
#include "runtime/backend.h"

namespace mini_infer {
    enum class OrtExecutionProvider {
        CPU,
        CUDA,
        TENSORRT,
    };

    class OrtBackend final : public Backend {
    public:
        OrtBackend(std::string model_path, bool use_cuda = false, int device_id = 0);
        OrtBackend(std::string model_path, OrtExecutionProvider provider, int device_id = 0);
        ~OrtBackend() override;   // 显式声明
        std::string name() const override;
        bool init() override;
        std::vector<Tensor> run_many(const std::vector<Tensor>& inputs) override;
    private:
        std::string model_path_;
        OrtExecutionProvider provider_{OrtExecutionProvider::CPU};
        int device_id_{0};
        // PIMPL: 避免头文件暴露 ORT 细节
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    } 
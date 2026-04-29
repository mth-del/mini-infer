#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/backend.h"

namespace mini_infer {

class TensorRtBackend final : public Backend {
public:
    TensorRtBackend(std::string model_path, int device_id = 0);
    ~TensorRtBackend() override;

    std::string name() const override { return "tensorrt-native"; }
    bool init() override;
    std::vector<Tensor> run_many(const std::vector<Tensor>& inputs) override;

private:
    std::string model_path_;
    int device_id_{0};
    bool ready_{false};

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mini_infer

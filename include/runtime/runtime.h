#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/backend.h"
#include "runtime/tensor.h"

namespace mini_infer {

class Runtime {
public:
    bool load_model(const std::string& model_path);
    void set_backend(std::shared_ptr<Backend> backend);
    Tensor infer(const std::vector<Tensor>& inputs);
    std::vector<Tensor> infer_many(const std::vector<Tensor>& inputs);

private:
    std::string model_path_;
    std::shared_ptr<Backend> backend_;
};

}  // namespace mini_infer

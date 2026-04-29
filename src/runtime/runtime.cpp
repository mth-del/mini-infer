#include "runtime/runtime.h"

#include <stdexcept>

namespace mini_infer {

bool Runtime::load_model(const std::string& model_path) {
    model_path_ = model_path;
    return !model_path_.empty();
}

void Runtime::set_backend(std::shared_ptr<Backend> backend) {
    backend_ = std::move(backend);
}

Tensor Runtime::infer(const std::vector<Tensor>& inputs) {
    if (!backend_) {
        throw std::runtime_error("Backend is not set");
    }
    return backend_->run(inputs);
}

}  // namespace mini_infer

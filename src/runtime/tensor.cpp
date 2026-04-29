#include "runtime/tensor.h"

#include <cstddef>

namespace mini_infer {

std::size_t Tensor::numel() const {
    if (shape.empty()) {
        return data_size();
    }
    std::size_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<std::size_t>(d > 0 ? d : 0);
    }
    return n;
}

std::size_t Tensor::data_size() const {
    switch (elem_type) {
        case TensorElementType::FLOAT32:
        case TensorElementType::FLOAT16:
            return data.size();
        case TensorElementType::INT64:
            return int64_data.size();
    }
    return 0;
}

const char* ToString(TensorElementType type) {
    switch (type) {
        case TensorElementType::FLOAT32:
            return "float32";
        case TensorElementType::FLOAT16:
            return "float16";
        case TensorElementType::INT64:
            return "int64";
    }
    return "unknown";
}

}  // namespace mini_infer

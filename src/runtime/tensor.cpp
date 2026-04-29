#include "runtime/tensor.h"

#include <cstddef>

namespace mini_infer {

std::size_t Tensor::numel() const {
    if (shape.empty()) {
        return data.size();
    }
    std::size_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<std::size_t>(d > 0 ? d : 0);
    }
    return n;
}

}  // namespace mini_infer

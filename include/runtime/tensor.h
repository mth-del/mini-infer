#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mini_infer {

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;

    std::size_t numel() const;
};

}  // namespace mini_infer

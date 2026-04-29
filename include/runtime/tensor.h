#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mini_infer {

enum class TensorElementType {
    FLOAT32,
    FLOAT16,
    INT64,
};

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    TensorElementType elem_type{TensorElementType::FLOAT32};
    std::vector<float> data;
    std::vector<int64_t> int64_data;

    std::size_t numel() const;
    std::size_t data_size() const;
};

const char* ToString(TensorElementType type);

}  // namespace mini_infer

#pragma once

#include <string>
#include <vector>

#include "runtime/tensor.h"

namespace mini_infer {

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;
    virtual bool init() = 0;
    virtual std::vector<Tensor> run_many(const std::vector<Tensor>& inputs) = 0;

    virtual Tensor run(const std::vector<Tensor>& inputs) {
        auto outputs = run_many(inputs);
        return outputs.empty() ? Tensor{} : std::move(outputs.front());
    }
};

}  // namespace mini_infer

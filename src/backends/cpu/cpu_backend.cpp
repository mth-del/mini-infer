#include "runtime/backend.h"

namespace mini_infer {

class CpuBackend final : public Backend {
public:
    std::string name() const override { return "cpu"; }
    bool init() override { return true; }

    Tensor run(const std::vector<Tensor>& inputs) override {
        if (inputs.empty()) {
            return {};
        }
        return inputs.front();
    }
};

}  // namespace mini_infer

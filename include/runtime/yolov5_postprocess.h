#pragma once

#include <vector>

#include "runtime/tensor.h"
#include "runtime/yolo_types.h"

namespace mini_infer {

struct LetterboxMeta {
    float scale{1.0F};
    float pad_w{0.0F};
    float pad_h{0.0F};
    int orig_w{0};
    int orig_h{0};
};

YoloV5Output DecodeYoloV5Output(
    const Tensor& output,
    const LetterboxMeta& meta,
    float conf_thres = 0.25F,
    float nms_thres = 0.45F);

}  // namespace mini_infer

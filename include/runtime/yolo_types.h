// include/runtime/yolo_types.h
#pragma once
#include <vector>

namespace mini_infer {

struct Detection {
    float x1{0}, y1{0}, x2{0}, y2{0};
    float score{0};
    int class_id{-1};
};

struct YoloV5Output {
    std::vector<Detection> dets;
    int image_w{0};
    int image_h{0};
};

}  // namespace mini_infer
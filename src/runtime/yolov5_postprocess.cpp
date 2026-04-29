#include "runtime/yolov5_postprocess.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_infer {
namespace {

float Clamp(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

std::pair<float, int> ArgmaxCls(const float* cls_ptr, int cls_count) {
    float best_conf = 0.0F;
    int best_id = -1;
    for (int i = 0; i < cls_count; ++i) {
        if (cls_ptr[i] > best_conf) {
            best_conf = cls_ptr[i];
            best_id = i;
        }
    }
    return {best_conf, best_id};
}

void UndoLetterbox(
    float& x1,
    float& y1,
    float& x2,
    float& y2,
    const LetterboxMeta& meta) {
    const float inv = (meta.scale > 0.0F) ? (1.0F / meta.scale) : 1.0F;
    x1 = (x1 - meta.pad_w) * inv;
    y1 = (y1 - meta.pad_h) * inv;
    x2 = (x2 - meta.pad_w) * inv;
    y2 = (y2 - meta.pad_h) * inv;

    x1 = Clamp(x1, 0.0F, static_cast<float>(meta.orig_w - 1));
    y1 = Clamp(y1, 0.0F, static_cast<float>(meta.orig_h - 1));
    x2 = Clamp(x2, 0.0F, static_cast<float>(meta.orig_w - 1));
    y2 = Clamp(y2, 0.0F, static_cast<float>(meta.orig_h - 1));
}

float IoU(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = std::max(0.0F, ix2 - ix1);
    const float ih = std::max(0.0F, iy2 - iy1);
    const float inter = iw * ih;

    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0F) {
        return 0.0F;
    }
    return inter / uni;
}

void ClassWiseNms(
    const std::vector<Detection>& candidates,
    float nms_thres,
    std::vector<Detection>* out) {
    std::unordered_map<int, std::vector<Detection>> by_class;
    by_class.reserve(80);

    for (const auto& d : candidates) {
        by_class[d.class_id].push_back(d);
    }

    for (auto& kv : by_class) {
        auto& dets = kv.second;
        std::sort(
            dets.begin(),
            dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });

        std::vector<char> removed(dets.size(), 0);
        for (std::size_t i = 0; i < dets.size(); ++i) {
            if (removed[i]) {
                continue;
            }
            out->push_back(dets[i]);
            for (std::size_t j = i + 1; j < dets.size(); ++j) {
                if (removed[j]) {
                    continue;
                }
                if (IoU(dets[i], dets[j]) > nms_thres) {
                    removed[j] = 1;
                }
            }
        }
    }
}

}  // namespace

// output
YoloV5Output DecodeYoloV5Output(
    const Tensor& output,
    const LetterboxMeta& meta,
    float conf_thres,
    float nms_thres) {
    YoloV5Output res;
    res.image_w = meta.orig_w;
    res.image_h = meta.orig_h;

    if (output.shape.size() != 3 || output.shape[0] != 1) {
        return res;
    }

    const int64_t num_boxes = output.shape[1];
    const int64_t num_attrs = output.shape[2];
    if (num_boxes <= 0 || num_attrs < 6) {
        return res;
    }

    const int cls_count = static_cast<int>(num_attrs - 5);
    std::vector<Detection> candidates;
    candidates.reserve(static_cast<std::size_t>(num_boxes));

    for (int64_t i = 0; i < num_boxes; ++i) {
        const std::size_t base = static_cast<std::size_t>(i * num_attrs);
        if (base + static_cast<std::size_t>(num_attrs) > output.data.size()) {
            break;
        }

        const float cx = output.data[base + 0];
        const float cy = output.data[base + 1];
        const float w = output.data[base + 2];
        const float h = output.data[base + 3];
        const float obj = output.data[base + 4];

        auto [cls_conf, cls_id] = ArgmaxCls(&output.data[base + 5], cls_count);
        const float score = obj * cls_conf;
        if (score < conf_thres || cls_id < 0) {
            continue;
        }

        float x1 = cx - w * 0.5F;
        float y1 = cy - h * 0.5F;
        float x2 = cx + w * 0.5F;
        float y2 = cy + h * 0.5F;

        UndoLetterbox(x1, y1, x2, y2, meta);
        candidates.push_back({x1, y1, x2, y2, score, cls_id});
    }

    ClassWiseNms(candidates, nms_thres, &res.dets);
    std::sort(
        res.dets.begin(),
        res.dets.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });
    return res;
}

}  // namespace mini_infer

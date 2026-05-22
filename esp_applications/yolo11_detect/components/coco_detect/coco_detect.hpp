#pragma once
#include "dl_detect_base.hpp"

namespace coco_detect {
class Yolo11n : public dl::detect::DetectImpl {
public:
    static constexpr float default_score_thr = 0.25;
    static constexpr float default_nms_thr = 0.7;
    Yolo11n(const char *model_name, float score_thr, float nms_thr);

    // Override to enable dual-core inference
    std::list<dl::detect::result_t> &run(const dl::image::img_t &img) override;
};
} // namespace coco_detect

class COCODetect : public dl::detect::DetectWrapper {
public:
    typedef enum {
        YOLO11N_S8_V1,
        YOLO11N_S8_V2,
        YOLO11N_S8_V3,
        YOLO11N_320_S8_V3,
    } model_type_t;

    COCODetect(model_type_t model_type = YOLO11N_S8_V1, bool lazy_load = true);

private:
    void load_model() override;
    model_type_t m_model_type;
};
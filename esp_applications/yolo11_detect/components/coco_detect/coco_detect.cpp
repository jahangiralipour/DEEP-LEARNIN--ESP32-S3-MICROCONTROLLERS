#include "coco_detect.hpp"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dl_detect_yolo11_postprocessor.hpp"

extern const uint8_t coco_detect_espdl[] asm("_binary_coco_detect_espdl_start");
static const char *path = (const char *)coco_detect_espdl;

namespace coco_detect {

Yolo11n::Yolo11n(const char *model_name, float score_thr, float nms_thr)
{
    bool param_copy = (heap_caps_get_total_size(MALLOC_CAP_SPIRAM) >= 1024 * 1024 * 9);

    this->m_model = new dl::Model(
        path,
        model_name,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        param_copy
    );

    this->m_model->build(0, dl::MEMORY_MANAGER_GREEDY, false);

    this->m_image_preprocessor = new dl::image::ImagePreprocessor(
        this->m_model,
        {0.f, 0.f, 0.f},
        {255.f, 255.f, 255.f},
        false
    );
    this->m_image_preprocessor->enable_letterbox({114, 114, 114});

    this->m_postprocessor = new dl::detect::yolo11PostProcessor(
        this->m_model,
        this->m_image_preprocessor,
        score_thr,
        nms_thr,
        /*top_k=*/100,
        /*stages=*/{
            {8,  8,  0, 0},
            {16, 16, 0, 0},
            {32, 32, 0, 0},
        }
    );
}

std::list<dl::detect::result_t> &Yolo11n::run(const dl::image::img_t &img)
{
    m_image_preprocessor->preprocess(img);
    m_model->run(dl::RUNTIME_MODE_MULTI_CORE);
    m_postprocessor->clear_result();
    m_postprocessor->postprocess();
    return m_postprocessor->get_result(img.width, img.height);
}

} // namespace coco_detect

COCODetect::COCODetect(model_type_t model_type, bool lazy_load)
    : m_model_type(model_type)
{
    this->m_score_thr[0] = 0.25f;
    this->m_score_thr[1] = 0.25f;
    this->m_nms_thr[0]   = 0.70f;
    this->m_nms_thr[1]   = 0.70f;
    this->m_model         = nullptr;

    if (!lazy_load) {
        load_model();
    }
}

void COCODetect::load_model()
{
    if (this->m_model == nullptr) {
        ESP_LOGI("coco_detect", "Loading YOLO11n model from flash...");
        this->m_model = new coco_detect::Yolo11n(
            "coco_detect_yolo11n_s8_v1.espdl",
            this->m_score_thr[0],
            this->m_nms_thr[0]
        );
    }
}
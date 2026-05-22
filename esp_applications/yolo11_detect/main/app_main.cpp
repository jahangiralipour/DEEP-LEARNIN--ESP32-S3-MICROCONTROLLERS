#include "coco_detect.hpp"
#include "dl_image_jpeg.hpp"
#include "esp_log.h"

extern const uint8_t testpic_jpg_start[] asm("_binary_testpic_jpg_start");
extern const uint8_t testpic_jpg_end[] asm("_binary_testpic_jpg_end");
const char *TAG = "yolo11n";

extern "C" void app_main(void)
{
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void *)testpic_jpg_start, 
        .data_len = (size_t)(testpic_jpg_end - testpic_jpg_start)
    };
    
    ESP_LOGI(TAG, "Decoding sample JPEG image...");
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    // Pass false to 'lazy_load' to force immediate model allocation safely inside the constructor
    ESP_LOGI(TAG, "Initializing detector...");
    COCODetect *detect = new COCODetect(COCODetect::YOLO11N_S8_V1, false);

    ESP_LOGI(TAG, "Running deep learning inference pipeline...");
    auto &detect_results = detect->run(img);
    
    ESP_LOGI(TAG, "Inference completed! Parsing results:");
    for (const auto &res : detect_results) {
        ESP_LOGI(TAG,
                 "[category: %d, score: %f, x1: %d, y1: %d, x2: %d, y2: %d]",
                 res.category, res.score, res.box[0], res.box[1], res.box[2], res.box[3]);
    }
}
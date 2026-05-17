#include <stdio.h>
#include <vector>
#include <cmath>
#include "dl_image_jpeg.hpp"
#include "esp_log.h"

// Core model and tensor implementation wrappers
#include "dl_model_base.hpp" 
#include "dl_tensor_base.hpp"

// Binary symbol pointers linked automatically via CMake
extern const uint8_t packed_model_espdl_start[] asm("_binary_packed_model_espdl_start");
extern const uint8_t packed_model_espdl_end[]   asm("_binary_packed_model_espdl_end");

extern const uint8_t cat_jpg_start[] asm("_binary_cat_jpg_start");
extern const uint8_t cat_jpg_end[]   asm("_binary_cat_jpg_end");

const char *TAG = "mobilenetv2_cls";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting MobileNetV2 Inference pipeline...");

    size_t model_size = (size_t)(packed_model_espdl_end - packed_model_espdl_start);
    ESP_LOGI(TAG, "Embedded Model Address: %p, Size: %d bytes", packed_model_espdl_start, (int)model_size);

    if (model_size == 0) {
        ESP_LOGE(TAG, "Model missing from designated Flash mapping space!");
        return;
    }

    // Decode image
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void *)cat_jpg_start, 
        .data_len = (size_t)(cat_jpg_end - cat_jpg_start)
    };
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    ESP_LOGI(TAG, "Image decoded. Width: %d, Height: %d", img.width, img.height);

    // Setup Model specifying its location in Flash memory
    dl::Model *model = new dl::Model((const char*)packed_model_espdl_start,
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                     0, 
                                     dl::MEMORY_MANAGER_GREEDY);

    // Get the exact properties required by the model's input layer dynamically
    auto model_inputs = model->get_inputs();
    int target_exponent = model_inputs.begin()->second->get_exponent();
    ESP_LOGI(TAG, "Model expects input tensor with quantization exponent: %d", target_exponent);

    // Convert pixel array from Unsigned UINT8 to Signed INT8 (-128 to 127 optimization layout)
    int total_pixels = img.width * img.height * 3;
    int8_t *int8_img_data = (int8_t *)heap_caps_malloc(total_pixels * sizeof(int8_t), MALLOC_CAP_SPIRAM);
    if (int8_img_data == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate input buffer in PSRAM!");
        delete model;
        heap_caps_free(img.data);
        return;
    }
    
    uint8_t *raw_pixels = (uint8_t *)img.data;
    for (int i = 0; i < total_pixels; ++i) {
        int8_img_data[i] = (int8_t)((int)raw_pixels[i] - 128); 
    }

    // Define configuration values for the input tensor matching MobileNetV2 [Height, Width, Channels]
    std::vector<int> input_shape = {img.height, img.width, 3}; 

    // Instantiate TensorBase matching the expected INT8 footprint and the parsed exponent
    dl::TensorBase input_tensor(input_shape,               // shape
                                (const void*)int8_img_data,// element data pointer (Signed INT8 in PSRAM)
                                target_exponent,           // dynamic exponent matched to calibration
                                dl::DATA_TYPE_INT8,        // Data type set to signed INT8
                                false);                    // deep = false (Directly references allocated pointer)

    // Process inference via the runner graph passing our input tensor pointer
    ESP_LOGI(TAG, "Executing model graph inference...");
    model->run(&input_tensor); 

    // Safely extract output maps managed by the model execution state
    auto &outputs = model->get_outputs();
    auto &result_tensor = outputs.begin()->second; 
    
    int num_classes = result_tensor->get_shape().back();
    std::vector<float> float_logits(num_classes, 0.0f);

    // Extract raw scores and handle low-level dequantization depending on output data type
    if (result_tensor->get_dtype() == dl::DATA_TYPE_INT8 || result_tensor->get_dtype() == dl::DATA_TYPE_INT32) {
        int target_out_exponent = result_tensor->get_exponent();
        float scale = std::pow(2.0f, target_out_exponent);

        if (result_tensor->get_dtype() == dl::DATA_TYPE_INT8) {
            int8_t *quant_scores = result_tensor->get_element_ptr<int8_t>();
            for (int i = 0; i < num_classes; ++i) {
                float_logits[i] = (float)quant_scores[i] * scale;
            }
        } else {
            int32_t *quant_scores = result_tensor->get_element_ptr<int32_t>();
            for (int i = 0; i < num_classes; ++i) {
                float_logits[i] = (float)quant_scores[i] * scale;
            }
        }
    } else {
        // Output layer data is already mapped to raw float logits
        float *raw_logits = result_tensor->get_element_ptr<float>();
        for (int i = 0; i < num_classes; ++i) {
            float_logits[i] = raw_logits[i];
        }
    }

    // --- Compute Mathematically Stable Softmax Normalization ---
    // 1. Locate the maximum logit value across the distribution array
    float max_logit = float_logits[0];
    for (int i = 1; i < num_classes; ++i) {
        if (float_logits[i] > max_logit) {
            max_logit = float_logits[i];
        }
    }

    // 2. Perform exponentiation shifted by max_logit to mitigate potential floating-point overflow
    float sum_exp = 0.0f;
    std::vector<float> probabilities(num_classes, 0.0f);
    for (int i = 0; i < num_classes; ++i) {
        probabilities[i] = std::exp(float_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    // 3. Normalize the final score array into uniform 0.0 - 1.0 probability space
    int top_class = -1;
    float max_prob = -1.0f;
    for (int i = 0; i < num_classes; ++i) {
        probabilities[i] /= sum_exp;
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            top_class = i;
        }
    }

    ESP_LOGI(TAG, "--- Top Detection Metrics ---");
    if (top_class != -1) {
        ESP_LOGI(TAG, "Predicted Class Index: %d, Confidence Score: %.4f (%.2f%%)", 
                 top_class, max_prob, max_prob * 100.0f);
    } else {
        ESP_LOGW(TAG, "No valid classification metrics found.");
    }

    // Free resources
    delete model;
    heap_caps_free(img.data);
    heap_caps_free(int8_img_data);
    ESP_LOGI(TAG, "Inference completed successfully.");
}
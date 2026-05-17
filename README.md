# DEEP-LEARNIN--ESP32-S3-MICROCONTROLLERS

# Project Title: EMPIRICAL EVALUATION OF DEEP LEARNING INFERENCE ON ESP32-S3 MICROCONTROLLERS
### Group Members: Jahangir Alipournajmi Iranagh - HAKAN YÖRÜK

## Project Description
This study focuses on the implementation and empirical evaluation of lightweight deep learning vision models on the resource-constrained **ESP32-S3 microcontroller** platform. Rather than using live camera streaming, this framework utilizes pre-collected datasets to strictly benchmark model capabilities under embedded hardware limitations.

The project evaluates two distinct vision tasks:
*   **Image Classification:** Benchmarking a lightweight **MobileNetV2** model.
*   **Object Detection:** Benchmarking a compact **YOLO**-based model (e.g., YOLO11n).

### Key Objectives
The primary goal of this repository is to analyze the critical trade-offs inherent to TinyML deployments:
*   **Accuracy vs. Inference Latency:** Measuring how processing speed impacts performance.
*   **Model Size & Memory Footprint:** Evaluating RAM and Flash utilization under rigid embedded constraints.
*   **Optimization Pipeline:** Documenting the model transformation pipeline (converting models to ONNX and applying quantization to fit the microcontroller).

The findings of this study establish a foundational benchmark for real-time TinyML applications on the ESP32 platform.


## How to Run the Project


### Prerequisites
### 1. Host Machine (Python Environment)
*   **Python:** Version `3.10` or higher.
*   **Core Libraries:**
    *   `torch` & `torchvision` (For model definition, dataset loading, and transforms)
*   **Quantization Engine:**
    *   `ppq` (PPL Quantization Tool core library)
    *   `esp_ppq` (Espressif's custom PPQ extension package for target hardware graph generation)
*   **Calibration Dataset Utilities:**
    *   `urllib` & `zipfile` (For automated dataset fetching and decompression)
    *   ImageNet or a compatible custom calibration dataset handled via `datasets.ImageFolder`
  ### 2. Embedded Target (ESP32-S3 Environment)
*   **ESP-IDF v5.1 or higher:** The official Espressif IoT Development Framework must be installed and configured in your system path.
*   **ESP-DL Library:** Espressif's Deep Learning library for loading the optimized model graph and executing INT8 quantized inference.
*   **Hardware:** An **ESP32-S3** development board (e.g., ESP32-S3-WROOM-1).
*   **Build Tools:** `CMake` (v3.16+) and `Ninja` build system (both are automatically included if using the standard ESP-IDF desktop installer or VS Code extension).

### Installation & Setup


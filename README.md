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


### Installation & System Setup (mobilenetv2_cls project)

#### Step 1: Clone the Project & Set Up the Workspace Environment
First, clone this project repository to your local machine. Ensure you also have the official Espressif Deep Learning framework (`ESP-DL`) cloned on your system:

```bash
# Clone this project repository
git clone <https://github.com/jahangiralipour/DEEP-LEARNIN--ESP32-S3-MICROCONTROLLERS>
cd DEEP-LEARNIN--ESP32-S3-MICROCONTROLLERS


 #Ensure you have ESP-DL downloaded on your machine (e.g., in your shared F:/esp/ directory)
# git clone --recursive [https://github.com/espressif/esp-dl.git](https://github.com/espressif/esp-dl.git)

```
#### Step 2: Register the ESP-DL Path in CMake
To ensure the build toolchain successfully resolves and compiles the neural network execution layers (`dl::Model`, `dl::TensorBase`), configure the `CMakeLists.txt` file located directly inside the `mobilenetv2_cls` folder to link your local `esp-dl` library path.

Open `mobilenetv2_cls/CMakeLists.txt` and verify or update the path to match your local system environment. Replace the placeholder path below with the **exact absolute path where ESP-DL is installed on your machine**. Note that you should point directly to the **root** folder of the downloaded `esp-dl` repository—the ESP-IDF build system will automatically look inside it for the required `components` folder:

```cmake
cmake_minimum_required(VERSION 3.16)

# Provide the absolute path to your downloaded root esp-dl repository on your machine
set(EXTRA_COMPONENT_DIRS "C:/YOUR_LOCAL_PATH_TO/esp-dl")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(mobilenetv2_cls)
```

#### Step 3: Configure Hardware Peripheral Options via menuconfig
Because executing deep learning models requires significant memory overhead, you must explicitly enable and configure **External Octal PSRAM** via the ESP-IDF interactive terminal configuration interface.

1. Open your terminal, navigate to the `mobilenetv2_cls` directory, ensure your ESP-IDF environment is exported, and launch the menu configuration utility:
   ```bash
   cd mobilenetv2_cls
   idf.py menuconfig
   ```

2.Enable External SPI RAM Layout:Navigate to: Component config $\rightarrow$ High Performance Embedded Wi-Fi / Bluetooth (ESP32-S3) $\rightarrow$ SPI RAM configHighlight Support for external, SPI-connected RAM and press Y to enable it.Ensure SPI RAM access method is set to: Make RAM allocatable using heap_caps_malloc().
3.Align Bus Speed & Octal OPI Hardware Profiles:

Navigate back up to the main menu screen and enter: Serial flasher config

Change Flash SPI mode explicitly to OPI.

Navigate back down to SPI RAM config, find Type of SPI RAM Chip (or SPI RAM Mode), and change it to Octal SPI PSRAM / OPI Mode.

Match the operating frequencies: verify both Flash Speed and SPI RAM Speed are set symmetrically to 80MHz.
4.Save & Exit:

Press S to write and save the configuration changes to your local sdkconfig file.

Press Q to exit the setup utility.

#### Step 4: Build, Flash, and Inference Execution
Since changing low-level peripheral memory definitions alters early bootloader configurations, always clear the compilation cache fully before rebuilding your binary workspace.

Execute the following sequential terminal commands to clean, compile, deploy, and inspect the real-time runtime serial monitor output hooks:
 Wipe old configuration caches completely
 ```
idf.py fullclean
```

 Build the second-stage bootloader and the target inference application binary structures
 ```
idf.py build
```

 Flash the compiled binary image payload onto your ESP32-S3 board over the assigned serial interface
 and attach the monitoring stream immediately to view the raw logs.
 ```
idf.py flash monitor
```

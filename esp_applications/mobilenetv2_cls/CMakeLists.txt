cmake_minimum_required(VERSION 3.16)

# Provide the absolute path to your downloaded esp-dl repository
set(EXTRA_COMPONENT_DIRS "F:/esp/esp-dl")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(mobilenetv2_cls)
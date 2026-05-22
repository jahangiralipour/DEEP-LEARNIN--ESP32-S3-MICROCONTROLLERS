idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

set(src_dirs        .)
set(include_dirs    .)
set(requires        esp-dl)

set(packed_model ${BUILD_DIR}/espdl_models/coco_detect.espdl)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

# Hardcode the target directly to your single model file
set(single_model ${COMPONENT_DIR}/models/s3/yolo11n_s8.espdl)

file(MAKE_DIRECTORY ${BUILD_DIR}/espdl_models)

# Create the explicit command to copy and establish the standard output file name
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Deploying YOLO11 model to build directory..."
    COMMAND python -c "import shutil; shutil.copy(r'${single_model}', r'${packed_model}')"
    DEPENDS ${single_model}
    VERBATIM)

# Add aligned binary under the default name expected by dl::detect::DetectWrapper
target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
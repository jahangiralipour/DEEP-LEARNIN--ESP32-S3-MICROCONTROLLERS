# 1. Locate and pull in the esp-dl component's packing utilities
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# 2. Define source and include paths
set(src_dirs        .)
set(include_dirs    .)

# 3. Define requirements (Bypassing the missing external BSP components)
set(requires        esp-dl esp_lcd)
set(embed_files     "cat.jpg")

# Register the main component first so ${COMPONENT_LIB} becomes available
idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} EMBED_FILES ${embed_files})

# 4. Handle the .espdl model packing directly from your local folders
set(packed_model ${BUILD_DIR}/main_models/packed_model.espdl)
file(MAKE_DIRECTORY ${BUILD_DIR}/main_models)

# Resolve target-specific path inside your local main folder
if (IDF_TARGET STREQUAL "esp32s3")
    set(models_dir ${COMPONENT_DIR}/models/esp32s3)
elseif (IDF_TARGET STREQUAL "esp32p4")
    set(models_dir ${COMPONENT_DIR}/models/esp32p4)
endif()

# Point to your local model file location
set(my_model ${models_dir}/imagenet_cls_mobilenetv2_s8_v1.espdl)

# Run the packaging script provided by esp-dl automatically on compilation
set(pack_model_exe ${espdl_dir}/fbs_loader/pack_espdl_models.py)
add_custom_command(
    OUTPUT ${packed_model}
    COMMENT "Packing local project ESPDL model..."
    COMMAND python ${pack_model_exe} --model_path ${my_model} --out_file ${packed_model}
    DEPENDS ${my_model}
    VERBATIM
)

# Enforce build order sequence
add_custom_target(pack_my_model DEPENDS ${packed_model})
add_dependencies(${COMPONENT_LIB} pack_my_model)

# 5. Link the packed binary model directly to your main component's flash memory (rodata)
target_add_aligned_binary_data(${COMPONENT_LIB} ${packed_model} BINARY)
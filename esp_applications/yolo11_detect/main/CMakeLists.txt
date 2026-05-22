set(src_dirs        ./)
set(include_dirs    ./)

# Only require your local neural network component, removing the missing BSP
set(requires        coco_detect)

# Define the static image asset embedded into flash
set(embed_files     "testpic.jpg")

idf_component_register(SRC_DIRS ${src_dirs} 
                       INCLUDE_DIRS ${include_dirs} 
                       REQUIRES ${requires} 
                       EMBED_FILES ${embed_files})
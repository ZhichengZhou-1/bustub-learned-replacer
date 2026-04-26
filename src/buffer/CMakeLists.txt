add_library(
        bustub_buffer
        OBJECT
        buffer_pool_manager.cpp
        arc_replacer.cpp
        clock_replacer.cpp
        lru_replacer.cpp
        lru_k_replacer.cpp
        learned_replacer.cpp)

# ONNX Runtime
target_include_directories(bustub_buffer PRIVATE /opt/homebrew/Cellar/onnxruntime/1.25.0/include/onnxruntime)
target_link_libraries(bustub_buffer PRIVATE /opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:bustub_buffer>
        PARENT_SCOPE)

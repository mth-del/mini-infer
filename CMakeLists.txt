cmake_minimum_required(VERSION 3.16)
project(mini_infer LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/onnx")

# ------------------------------------------------------------------------------
# Build options
# ------------------------------------------------------------------------------
option(MINI_INFER_ENABLE_TENSORRT "Build TensorRT backend stubs" ON)
option(MINI_INFER_ENABLE_OPENVINO "Build OpenVINO backend stubs" OFF)
option(MINI_INFER_ENABLE_ONNXRUNTIME "Build ONNX Runtime backend" ON)

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------
add_library(mini_infer_core
    src/runtime/runtime.cpp
    src/runtime/tensor.cpp
    src/runtime/qwen_tokenizer.cpp
    src/runtime/yolov5_postprocess.cpp
    src/graph/graph.cpp
    src/memory/allocator.cpp
    src/backends/cpu/cpu_backend.cpp
)

target_include_directories(mini_infer_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mini_infer_core PRIVATE dl)

# Optional backend sources
if(MINI_INFER_ENABLE_TENSORRT)
    target_sources(mini_infer_core PRIVATE src/backends/tensorrt/tensorrt_backend.cpp)
    target_compile_definitions(mini_infer_core PUBLIC MINI_INFER_HAS_TENSORRT=1)
endif()

if(MINI_INFER_ENABLE_OPENVINO)
    target_sources(mini_infer_core PRIVATE src/backends/openvino/openvino_backend.cpp)
    target_compile_definitions(mini_infer_core PUBLIC MINI_INFER_HAS_OPENVINO=1)
endif()

if(MINI_INFER_ENABLE_ONNXRUNTIME)
    target_sources(mini_infer_core PRIVATE src/backends/onnx/ort_backend.cpp)
    target_include_directories(mini_infer_core PUBLIC ${ONNXRUNTIME_ROOT}/include)
    target_link_libraries(mini_infer_core PRIVATE
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    )
    if(MINI_INFER_ENABLE_ORT_CUDA)
        target_compile_definitions(mini_infer_core PUBLIC MINI_INFER_ENABLE_ORT_CUDA=1)
    endif()
endif()

# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------
add_executable(infer_cli tools/infer_cli.cpp)
target_link_libraries(infer_cli PRIVATE mini_infer_core)

if(MINI_INFER_ENABLE_ONNXRUNTIME)
    add_executable(onnx_model_info tools/onnx_model_info.cpp)
    target_include_directories(onnx_model_info PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_libraries(onnx_model_info PRIVATE
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    )
    add_executable(qwen_cli_cpp tools/qwen_cli.cpp)
    target_link_libraries(qwen_cli_cpp PRIVATE mini_infer_core)

    set_target_properties(infer_cli PROPERTIES
        BUILD_RPATH "\$ORIGIN/../src/backends/onnx/lib"
        INSTALL_RPATH "\$ORIGIN/../src/backends/onnx/lib"
    )
    set_target_properties(onnx_model_info PROPERTIES
        BUILD_RPATH "\$ORIGIN/../src/backends/onnx/lib"
        INSTALL_RPATH "\$ORIGIN/../src/backends/onnx/lib"
    )
    set_target_properties(qwen_cli_cpp PROPERTIES
        BUILD_RPATH "\$ORIGIN/../src/backends/onnx/lib"
        INSTALL_RPATH "\$ORIGIN/../src/backends/onnx/lib"
    )
endif()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
enable_testing()
add_executable(test_smoke tests/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE mini_infer_core)
add_test(NAME smoke COMMAND test_smoke)
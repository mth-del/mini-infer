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
set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT SDK root directory")

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
    src/runtime/llm_config.cpp
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

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS
            ${TENSORRT_ROOT}
            ${TENSORRT_ROOT}/include
            /usr/include
            /usr/include/x86_64-linux-gnu
            /usr/local/tensorrt/include
    )
    find_library(TENSORRT_NVINFER_LIBRARY nvinfer
        HINTS
            ${TENSORRT_ROOT}
            ${TENSORRT_ROOT}/lib
            ${TENSORRT_ROOT}/lib64
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/tensorrt/lib
    )
    find_library(TENSORRT_NVONNXPARSER_LIBRARY nvonnxparser
        HINTS
            ${TENSORRT_ROOT}
            ${TENSORRT_ROOT}/lib
            ${TENSORRT_ROOT}/lib64
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/tensorrt/lib
    )
    find_library(CUDART_LIBRARY cudart
        HINTS
            /usr/local/cuda/lib64
            /usr/local/cuda-12.8/lib64
            /usr/local/cuda-12.8/targets/x86_64-linux/lib
            /usr/lib/x86_64-linux-gnu
    )
    find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
        HINTS
            /usr/local/cuda/include
            /usr/local/cuda-12.8/include
            /usr/local/cuda-12.8/targets/x86_64-linux/include
            /usr/include
    )

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND CUDART_LIBRARY AND CUDART_INCLUDE_DIR)
        target_include_directories(mini_infer_core PRIVATE
            ${TENSORRT_INCLUDE_DIR}
            ${CUDART_INCLUDE_DIR}
        )
        target_link_libraries(mini_infer_core PRIVATE
            ${TENSORRT_NVINFER_LIBRARY}
            ${CUDART_LIBRARY}
        )
        if(TENSORRT_NVONNXPARSER_LIBRARY)
            target_link_libraries(mini_infer_core PRIVATE ${TENSORRT_NVONNXPARSER_LIBRARY})
            target_compile_definitions(mini_infer_core PRIVATE MINI_INFER_TENSORRT_HAS_ONNX_PARSER=1)
        endif()
        target_compile_definitions(mini_infer_core PRIVATE MINI_INFER_TENSORRT_NATIVE=1)
        message(STATUS "Native TensorRT backend enabled: ${TENSORRT_NVINFER_LIBRARY}")
    else()
        message(WARNING "Native TensorRT SDK not found. TensorRT backend will build as an unavailable stub.")
    endif()
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

    if(MINI_INFER_ENABLE_TENSORRT
       AND TENSORRT_INCLUDE_DIR
       AND TENSORRT_NVINFER_LIBRARY
       AND TENSORRT_NVONNXPARSER_LIBRARY
       AND CUDART_LIBRARY
       AND CUDART_INCLUDE_DIR)
        add_executable(trt_build_engine tools/trt_build_engine.cpp)
        target_include_directories(trt_build_engine PRIVATE
            ${TENSORRT_INCLUDE_DIR}
            ${CUDART_INCLUDE_DIR}
        )
        target_link_libraries(trt_build_engine PRIVATE
            ${TENSORRT_NVINFER_LIBRARY}
            ${TENSORRT_NVONNXPARSER_LIBRARY}
            ${CUDART_LIBRARY}
        )
    endif()

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
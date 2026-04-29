#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

std::string ElementTypeName(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        default:
            return "unknown(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

void PrintShape(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) {
            std::cout << ",";
        }
    }
    std::cout << "]";
}

void PrintTensorInfo(Ort::Session& session, bool input) {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t count = input ? session.GetInputCount() : session.GetOutputCount();
    std::cout << (input ? "Inputs" : "Outputs") << " (" << count << ")\n";

    for (std::size_t i = 0; i < count; ++i) {
        auto name = input
            ? session.GetInputNameAllocated(i, allocator)
            : session.GetOutputNameAllocated(i, allocator);
        auto type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << "  #" << i
                  << " name=" << name.get()
                  << " type=" << ElementTypeName(tensor_info.GetElementType())
                  << " shape=";
        PrintShape(tensor_info.GetShape());
        std::cout << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: onnx_model_info <model.onnx>\n";
        return 2;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mini_infer_model_info");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        Ort::Session session(env, argv[1], options);

        std::cout << "Model: " << argv[1] << "\n";
        PrintTensorInfo(session, true);
        PrintTensorInfo(session, false);
    } catch (const std::exception& e) {
        std::cerr << "Failed to inspect model: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

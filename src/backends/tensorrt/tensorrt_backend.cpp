#include "runtime/tensorrt_backend.h"

#include <dlfcn.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef MINI_INFER_TENSORRT_NATIVE
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

namespace mini_infer {
namespace {

#ifdef MINI_INFER_TENSORRT_NATIVE
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << "\n";
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const {
        delete ptr;
    }
};

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
struct IoTensorInfo {
    std::string name;
    bool is_input{false};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    nvinfer1::Dims engine_shape{};
};

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    DeviceAllocation(DeviceAllocation&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~DeviceAllocation() { reset(); }

    void allocate(std::size_t bytes) {
        reset();
        bytes_ = bytes;
        const std::size_t cuda_bytes = bytes == 0 ? 1 : bytes;
        cudaError_t status = cudaMalloc(&ptr_, cuda_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
        }
    }

    void* get() const { return ptr_; }
    std::size_t bytes() const { return bytes_; }

private:
    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        bytes_ = 0;
    }

    void* ptr_{nullptr};
    std::size_t bytes_{0};
};

void CudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
    }
}

std::string DimsToString(const nvinfer1::Dims& dims) {
    std::string text = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            text += ",";
        }
        text += std::to_string(dims.d[i]);
    }
    text += "]";
    return text;
}

std::vector<int64_t> DimsToShape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("TensorRT output has unresolved dynamic dimension");
        }
        shape.push_back(static_cast<int64_t>(dims.d[i]));
    }
    return shape;
}

nvinfer1::Dims ShapeToDims(const std::vector<int64_t>& shape) {
    if (shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        throw std::runtime_error("TensorRT input rank exceeds max dims");
    }
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0 || shape[i] > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("TensorRT input dimension is out of range");
        }
        dims.d[i] = static_cast<int32_t>(shape[i]);
    }
    return dims;
}

std::size_t ElementCount(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    std::size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("TensorRT shape contains negative dimension");
        }
        if (dim == 0) {
            return 0;
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::size_t DataTypeSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(__half);
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64:
            return sizeof(int64_t);
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}

const char* DataTypeName(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64:
            return "int64";
#endif
        default:
            return "unsupported";
    }
}

const Tensor* FindInputTensor(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, const Tensor*>& input_by_name,
    const IoTensorInfo& info,
    std::size_t fallback_index) {
    auto it = input_by_name.find(info.name);
    if (it != input_by_name.end()) {
        return it->second;
    }
    if (fallback_index < inputs.size() && inputs[fallback_index].name.empty()) {
        return &inputs[fallback_index];
    }
    return nullptr;
}

std::vector<char> BuildHostInputBuffer(const Tensor& input, nvinfer1::DataType dtype) {
    const std::size_t numel = input.numel();
    std::vector<char> bytes(numel * DataTypeSize(dtype));

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: {
            if (input.elem_type != TensorElementType::FLOAT32 || input.data.size() != numel) {
                throw std::runtime_error("TensorRT float32 input requires FLOAT32 tensor data");
            }
            std::memcpy(bytes.data(), input.data.data(), bytes.size());
            break;
        }
        case nvinfer1::DataType::kHALF: {
            if ((input.elem_type != TensorElementType::FLOAT16 &&
                 input.elem_type != TensorElementType::FLOAT32) ||
                input.data.size() != numel) {
                throw std::runtime_error("TensorRT float16 input requires FLOAT16/FLOAT32 tensor data");
            }
            auto* out = reinterpret_cast<__half*>(bytes.data());
            for (std::size_t i = 0; i < numel; ++i) {
                out[i] = __float2half(input.data[i]);
            }
            break;
        }
        case nvinfer1::DataType::kINT32: {
            if (input.elem_type != TensorElementType::INT64 || input.int64_data.size() != numel) {
                throw std::runtime_error("TensorRT int32 input requires INT64 tensor data for conversion");
            }
            auto* out = reinterpret_cast<int32_t*>(bytes.data());
            for (std::size_t i = 0; i < numel; ++i) {
                if (input.int64_data[i] < std::numeric_limits<int32_t>::min() ||
                    input.int64_data[i] > std::numeric_limits<int32_t>::max()) {
                    throw std::runtime_error("TensorRT int32 input value is out of range");
                }
                out[i] = static_cast<int32_t>(input.int64_data[i]);
            }
            break;
        }
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: {
            if (input.elem_type != TensorElementType::INT64 || input.int64_data.size() != numel) {
                throw std::runtime_error("TensorRT int64 input requires INT64 tensor data");
            }
            std::memcpy(bytes.data(), input.int64_data.data(), bytes.size());
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT input data type");
    }
    return bytes;
}

Tensor DecodeOutputTensor(const IoTensorInfo& info, const std::vector<int64_t>& shape, const std::vector<char>& bytes) {
    const std::size_t numel = ElementCount(shape);
    Tensor out;
    out.name = info.name;
    out.shape = shape;

    switch (info.dtype) {
        case nvinfer1::DataType::kFLOAT: {
            out.elem_type = TensorElementType::FLOAT32;
            const auto* data = reinterpret_cast<const float*>(bytes.data());
            if (numel > 0) {
                out.data.assign(data, data + numel);
            }
            break;
        }
        case nvinfer1::DataType::kHALF: {
            out.elem_type = TensorElementType::FLOAT16;
            const auto* data = reinterpret_cast<const __half*>(bytes.data());
            out.data.resize(numel);
            for (std::size_t i = 0; i < numel; ++i) {
                out.data[i] = __half2float(data[i]);
            }
            break;
        }
        case nvinfer1::DataType::kINT32: {
            out.elem_type = TensorElementType::INT64;
            const auto* data = reinterpret_cast<const int32_t*>(bytes.data());
            out.int64_data.reserve(numel);
            for (std::size_t i = 0; i < numel; ++i) {
                out.int64_data.push_back(static_cast<int64_t>(data[i]));
            }
            break;
        }
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: {
            out.elem_type = TensorElementType::INT64;
            const auto* data = reinterpret_cast<const int64_t*>(bytes.data());
            if (numel > 0) {
                out.int64_data.assign(data, data + numel);
            }
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT output data type");
    }
    return out;
}
#endif

std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("failed to open TensorRT engine: " + path);
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<std::size_t>(size));
    if (!ifs.read(data.data(), size)) {
        throw std::runtime_error("failed to read TensorRT engine: " + path);
    }
    return data;
}
#endif

}  // namespace

struct TensorRtBackend::Impl {
#ifdef MINI_INFER_TENSORRT_NATIVE
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context;
    cudaStream_t stream{nullptr};
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
    std::vector<IoTensorInfo> io_tensors;
    std::vector<IoTensorInfo> inputs;
    std::vector<IoTensorInfo> outputs;
#endif
#endif
};

TensorRtBackend::TensorRtBackend(std::string model_path, int device_id)
    : model_path_(std::move(model_path)), device_id_(device_id), impl_(std::make_unique<Impl>()) {}

TensorRtBackend::~TensorRtBackend() {
#ifdef MINI_INFER_TENSORRT_NATIVE
    if (impl_ && impl_->stream) {
        cudaStreamDestroy(impl_->stream);
        impl_->stream = nullptr;
    }
#endif
}

bool TensorRtBackend::init() {
#ifndef MINI_INFER_TENSORRT_NATIVE
    void* handle = dlopen("libnvinfer.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Native TensorRT backend unavailable: libnvinfer.so not found.\n";
        std::cerr << "Install TensorRT SDK/runtime and reconfigure CMake.\n";
        return false;
    }
    dlclose(handle);
    std::cerr << "Native TensorRT library is visible, but this build was configured without NvInfer.h.\n";
    std::cerr << "Set TENSORRT_ROOT and rerun CMake to enable MINI_INFER_TENSORRT_NATIVE.\n";
    return false;
#else
    try {
        const auto engine_bytes = ReadBinaryFile(model_path_);
        cudaError_t cuda_status = cudaSetDevice(device_id_);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_status));
        }
        cuda_status = cudaStreamCreate(&impl_->stream);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cuda_status));
        }

        impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
        if (!impl_->runtime) {
            throw std::runtime_error("createInferRuntime failed");
        }
        impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
            engine_bytes.data(), engine_bytes.size()));
        if (!impl_->engine) {
            throw std::runtime_error("deserializeCudaEngine failed");
        }
        impl_->context.reset(impl_->engine->createExecutionContext());
        if (!impl_->context) {
            throw std::runtime_error("createExecutionContext failed");
        }

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
        const int32_t tensor_count = impl_->engine->getNbIOTensors();
        impl_->io_tensors.reserve(static_cast<std::size_t>(tensor_count));
        for (int32_t i = 0; i < tensor_count; ++i) {
            const char* tensor_name = impl_->engine->getIOTensorName(i);
            if (!tensor_name) {
                throw std::runtime_error("TensorRT engine returned a null IO tensor name");
            }
            IoTensorInfo info;
            info.name = tensor_name;
            info.is_input =
                impl_->engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT;
            info.dtype = impl_->engine->getTensorDataType(tensor_name);
            info.engine_shape = impl_->engine->getTensorShape(tensor_name);
            std::cout << "TensorRT "
                      << (info.is_input ? "input" : "output")
                      << " name=" << info.name
                      << " dtype=" << DataTypeName(info.dtype)
                      << " shape=" << DimsToString(info.engine_shape)
                      << "\n";
            impl_->io_tensors.push_back(info);
            if (info.is_input) {
                impl_->inputs.push_back(info);
            } else {
                impl_->outputs.push_back(info);
            }
        }
#else
        throw std::runtime_error(
            "Native TensorRT execution requires TensorRT 8.5+ for named IO tensor APIs");
#endif

        ready_ = true;
        std::cout << "Native TensorRT backend initialized, device_id=" << device_id_
                  << ", engine=" << model_path_ << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Native TensorRT init failed: " << e.what() << "\n";
        ready_ = false;
        return false;
    }
#endif
}

std::vector<Tensor> TensorRtBackend::run_many(const std::vector<Tensor>& inputs) {
    if (!ready_) {
        throw std::runtime_error("TensorRtBackend is not initialized");
    }
    if (inputs.empty()) {
        return {};
    }
#ifndef MINI_INFER_TENSORRT_NATIVE
    throw std::runtime_error("Native TensorRT backend is not available in this build");
#else
#if NV_TENSORRT_MAJOR <= 8 && NV_TENSORRT_MINOR < 5
    throw std::runtime_error(
        "Native TensorRT execution requires TensorRT 8.5+ for named IO tensor APIs");
#else
    std::unordered_map<std::string, const Tensor*> input_by_name;
    for (const Tensor& input : inputs) {
        if (!input.name.empty()) {
            input_by_name[input.name] = &input;
        }
    }

    std::vector<std::vector<char>> host_input_buffers;
    std::vector<DeviceAllocation> device_inputs;
    host_input_buffers.reserve(impl_->inputs.size());
    device_inputs.reserve(impl_->inputs.size());

    for (std::size_t i = 0; i < impl_->inputs.size(); ++i) {
        const IoTensorInfo& info = impl_->inputs[i];
        const Tensor* input = FindInputTensor(inputs, input_by_name, info, i);
        if (!input) {
            throw std::runtime_error("missing TensorRT input tensor: " + info.name);
        }
        if (input->shape.empty()) {
            throw std::runtime_error("TensorRT input shape is empty: " + info.name);
        }

        const nvinfer1::Dims dims = ShapeToDims(input->shape);
        if (!impl_->context->setInputShape(info.name.c_str(), dims)) {
            throw std::runtime_error("TensorRT setInputShape failed for: " + info.name);
        }

        host_input_buffers.push_back(BuildHostInputBuffer(*input, info.dtype));
        device_inputs.emplace_back();
        device_inputs.back().allocate(host_input_buffers.back().size());
        if (!host_input_buffers.back().empty()) {
            CudaCheck(
                cudaMemcpyAsync(
                    device_inputs.back().get(),
                    host_input_buffers.back().data(),
                    host_input_buffers.back().size(),
                    cudaMemcpyHostToDevice,
                    impl_->stream),
                "cudaMemcpyAsync H2D");
        }
        if (!impl_->context->setTensorAddress(info.name.c_str(), device_inputs.back().get())) {
            throw std::runtime_error("TensorRT setTensorAddress failed for input: " + info.name);
        }
    }

    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::vector<char>> host_output_buffers;
    std::vector<DeviceAllocation> device_outputs;
    output_shapes.reserve(impl_->outputs.size());
    host_output_buffers.reserve(impl_->outputs.size());
    device_outputs.reserve(impl_->outputs.size());

    for (const IoTensorInfo& info : impl_->outputs) {
        const nvinfer1::Dims dims = impl_->context->getTensorShape(info.name.c_str());
        std::vector<int64_t> shape = DimsToShape(dims);
        const std::size_t bytes = ElementCount(shape) * DataTypeSize(info.dtype);
        output_shapes.push_back(std::move(shape));
        host_output_buffers.emplace_back(bytes);
        device_outputs.emplace_back();
        device_outputs.back().allocate(bytes);
        if (!impl_->context->setTensorAddress(info.name.c_str(), device_outputs.back().get())) {
            throw std::runtime_error("TensorRT setTensorAddress failed for output: " + info.name);
        }
    }

    if (!impl_->context->enqueueV3(impl_->stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
        if (!host_output_buffers[i].empty()) {
            CudaCheck(
                cudaMemcpyAsync(
                    host_output_buffers[i].data(),
                    device_outputs[i].get(),
                    host_output_buffers[i].size(),
                    cudaMemcpyDeviceToHost,
                    impl_->stream),
                "cudaMemcpyAsync D2H");
        }
    }
    CudaCheck(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");

    std::vector<Tensor> outputs;
    outputs.reserve(impl_->outputs.size());
    for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
        outputs.push_back(DecodeOutputTensor(
            impl_->outputs[i], output_shapes[i], host_output_buffers[i]));
    }
    return outputs;
#endif
#endif
}

}  // namespace mini_infer

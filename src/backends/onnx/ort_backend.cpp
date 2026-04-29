#include "ort_backend.h"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"

namespace mini_infer{

    struct OrtBackend::Impl{
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mini_infer"};
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<std::string> input_names_str;
        std::vector<std::string> output_names_str;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
    };

    OrtBackend::OrtBackend(std::string model_path, bool use_cuda, int device_id)
    : model_path_(std::move(model_path)), use_cuda_(use_cuda), device_id_(device_id) {}

    OrtBackend::~OrtBackend() = default;  // 放在 Impl 定义之后

    bool OrtBackend::init() {
        try {
            impl_ = std::make_unique<Impl>();
            impl_->session_options.SetIntraOpNumThreads(1);
            impl_->session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef MINI_INFER_ENABLE_ORT_CUDA
                if (use_cuda_) {
                    const OrtApi& api = Ort::GetApi();
                    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
                    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
                    std::string device_id_str = std::to_string(device_id_);
                    std::vector<const char*> keys{
                        "device_id",
                        "arena_extend_strategy",
                        "cudnn_conv_algo_search",
                        "do_copy_in_default_stream"
                    };
                    std::vector<const char*> values{
                        device_id_str.c_str(),
                        "kNextPowerOfTwo",
                        "EXHAUSTIVE",
                        "1"
                    };
                    std::string gpu_mem_limit;
                    if (const char* env_limit = std::getenv("MINI_INFER_ORT_CUDA_GPU_MEM_LIMIT")) {
                        gpu_mem_limit = env_limit;
                        keys.push_back("gpu_mem_limit");
                        values.push_back(gpu_mem_limit.c_str());
                    }
                    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
                        cuda_options, keys.data(), values.data(), keys.size()));
                    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                        impl_->session_options, cuda_options));
                    api.ReleaseCUDAProviderOptions(cuda_options);
                }
#endif
            impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path_.c_str(), impl_->session_options);
            Ort::AllocatorWithDefaultOptions allocator;
            const size_t input_count = impl_->session->GetInputCount();
            impl_->input_names_str.reserve(input_count);
            impl_->input_names.reserve(input_count);
            for (size_t i = 0; i < input_count; ++i) {
                auto name = impl_->session->GetInputNameAllocated(i, allocator);
                impl_->input_names_str.emplace_back(name.get());
            }
            for (auto& s : impl_->input_names_str) impl_->input_names.push_back(s.c_str());
            const size_t output_count = impl_->session->GetOutputCount();
            impl_->output_names_str.reserve(output_count);
            impl_->output_names.reserve(output_count);
            for (size_t i = 0; i < output_count; ++i) {
                auto name = impl_->session->GetOutputNameAllocated(i, allocator);
                impl_->output_names_str.emplace_back(name.get());
            }
            for (auto& s : impl_->output_names_str) impl_->output_names.push_back(s.c_str());
            return true;
        } catch (const std::exception& e) {
            std::cerr << "OrtBackend init failed: " << e.what() << std::endl;
            impl_.reset();
            return false;
        }
    }


    Tensor OrtBackend::run(const std::vector<Tensor>& inputs) {

        if (!impl_ || !impl_->session) {
            throw std::runtime_error("OrtBackend is not initialized");
        }
        if (inputs.empty()) {
            throw std::runtime_error("OrtBackend::run got empty inputs");
        }

        const Tensor& in = inputs[0];
        if (in.shape.empty() || in.data.empty()) {
            throw std::runtime_error("input tensor shape/data is empty");
        }
        auto input_type = impl_->session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType elem_type = input_tensor_info.GetElementType();

        std::cerr << "input_count=" << impl_->session->GetInputCount() << "\n";
        std::cerr << "input0_name=" << impl_->input_names_str[0] << "\n";
        std::cerr << "input0_elem_type=" << static_cast<int>(elem_type) << "\n";

        
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(1);
        
        // 需要让 fp16 buffer 生命周期覆盖 Run()
        std::vector<Ort::Float16_t> input_fp16;

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            Ort::Value x = Ort::Value::CreateTensor<float>(
                impl_->memory_info,
                const_cast<float*>(in.data.data()),
                in.data.size(),
                in.shape.data(),
                in.shape.size());
            ort_inputs.emplace_back(std::move(x));
        } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            input_fp16.resize(in.data.size());
            for (size_t i = 0; i < in.data.size(); ++i) {
                input_fp16[i] = Ort::Float16_t(in.data[i]);
            }
            Ort::Value x = Ort::Value::CreateTensor<Ort::Float16_t>(
                impl_->memory_info,
                input_fp16.data(),
                input_fp16.size(),
                in.shape.data(),
                in.shape.size());
            ort_inputs.emplace_back(std::move(x));
        } else {
            throw std::runtime_error("Unsupported model input element type");
        }
        
        auto ort_outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            impl_->output_names.data(),
            impl_->output_names.size());

        // 当前最小实现：取第一个输出
        Ort::Value& out0 = ort_outputs[0];
        if (!out0.IsTensor()) {
            throw std::runtime_error("ORT output[0] is not a tensor");
        }
        auto type_info = out0.GetTensorTypeAndShapeInfo();
        auto out_elem_type = type_info.GetElementType();
        std::vector<int64_t> out_shape = type_info.GetShape();
        size_t out_numel = type_info.GetElementCount();
        const float* out_data = out0.GetTensorData<float>();
        Tensor out;
        out.name = impl_->output_names_str.empty() ? "output0" : impl_->output_names_str[0];
        out.shape = std::move(out_shape);
        out.data.resize(out_numel);

        if (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* out_data = out0.GetTensorData<float>();
            out.data.assign(out_data, out_data + out_numel);
        } else if (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const Ort::Float16_t* out_data = out0.GetTensorData<Ort::Float16_t>();
            for (size_t i = 0; i < out_numel; ++i) {
                out.data[i] = static_cast<float>(out_data[i]);
            }
        } else {
            throw std::runtime_error("Unsupported model output element type");
        }
    

        return out;
    }
    

}
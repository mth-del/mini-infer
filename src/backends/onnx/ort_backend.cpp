#include "ort_backend.h"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    : OrtBackend(std::move(model_path), use_cuda ? OrtExecutionProvider::CUDA : OrtExecutionProvider::CPU, device_id) {}

    OrtBackend::OrtBackend(std::string model_path, OrtExecutionProvider provider, int device_id)
    : model_path_(std::move(model_path)), provider_(provider), device_id_(device_id) {}

    OrtBackend::~OrtBackend() = default;  // 放在 Impl 定义之后

    std::string OrtBackend::name() const {
        switch (provider_) {
            case OrtExecutionProvider::CPU:
                return "onnxruntime-cpu";
            case OrtExecutionProvider::CUDA:
                return "onnxruntime-cuda";
            case OrtExecutionProvider::TENSORRT:
                return "onnxruntime-tensorrt";
        }
        return "onnxruntime-unknown";
    }

    bool OrtBackend::init() {
        try {
            impl_ = std::make_unique<Impl>();
            impl_->session_options.SetIntraOpNumThreads(1);
            impl_->session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef MINI_INFER_ENABLE_ORT_CUDA
                if (provider_ == OrtExecutionProvider::CUDA) {
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


    std::vector<Tensor> OrtBackend::run_many(const std::vector<Tensor>& inputs) {

        if (!impl_ || !impl_->session) {
            throw std::runtime_error("OrtBackend is not initialized");
        }
        if (inputs.empty()) {
            throw std::runtime_error("OrtBackend::run got empty inputs");
        }

        std::unordered_map<std::string, const Tensor*> input_by_name;
        for (const Tensor& input : inputs) {
            if (!input.name.empty()) {
                input_by_name[input.name] = &input;
            }
        }

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(inputs.size());
        std::vector<const char*> run_input_names;
        run_input_names.reserve(inputs.size());

        // Temporary buffers must live until Session::Run() returns.
        std::vector<std::vector<Ort::Float16_t>> input_fp16_buffers;
        input_fp16_buffers.reserve(inputs.size());

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const std::string& model_input_name = i < impl_->input_names_str.size()
                ? impl_->input_names_str[i]
                : inputs[i].name;
            const Tensor* input = nullptr;
            if (!inputs[i].name.empty()) {
                input = &inputs[i];
            } else if (!model_input_name.empty()) {
                auto it = input_by_name.find(model_input_name);
                input = it == input_by_name.end() ? &inputs[i] : it->second;
            } else {
                input = &inputs[i];
            }

            if (!input || input->shape.empty()) {
                throw std::runtime_error("input tensor shape is empty");
            }
            // 支持长度为0的初始化KV cache
            if (input->data_size() == 0 && input->numel() != 0) {
                throw std::runtime_error("input tensor data is empty but shape is non-empty");
            }

            run_input_names.push_back(input->name.empty() ? model_input_name.c_str() : input->name.c_str());

            switch (input->elem_type) {
                case TensorElementType::FLOAT32: {
                    Ort::Value x = Ort::Value::CreateTensor<float>(
                        impl_->memory_info,
                        const_cast<float*>(input->data.data()),
                        input->data.size(),
                        input->shape.data(),
                        input->shape.size());
                    ort_inputs.emplace_back(std::move(x));
                    break;
                }
                case TensorElementType::FLOAT16: {
                    input_fp16_buffers.emplace_back(input->data.size());
                    auto& input_fp16 = input_fp16_buffers.back();
                    for (size_t j = 0; j < input->data.size(); ++j) {
                        input_fp16[j] = Ort::Float16_t(input->data[j]);
                    }
                    Ort::Value x = Ort::Value::CreateTensor<Ort::Float16_t>(
                        impl_->memory_info,
                        input_fp16.data(),
                        input_fp16.size(),
                        input->shape.data(),
                        input->shape.size());
                    ort_inputs.emplace_back(std::move(x));
                    break;
                }
                case TensorElementType::INT64: {
                    Ort::Value x = Ort::Value::CreateTensor<int64_t>(
                        impl_->memory_info,
                        const_cast<int64_t*>(input->int64_data.data()),
                        input->int64_data.size(),
                        input->shape.data(),
                        input->shape.size());
                    ort_inputs.emplace_back(std::move(x));
                    break;
                }
            }
        }
        
        auto ort_outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            run_input_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            impl_->output_names.data(),
            impl_->output_names.size());

        std::vector<Tensor> outputs;
        outputs.reserve(ort_outputs.size());
        for (std::size_t i = 0; i < ort_outputs.size(); ++i) {
            Ort::Value& ort_output = ort_outputs[i];
            if (!ort_output.IsTensor()) {
                throw std::runtime_error("ORT output is not a tensor");
            }

            auto type_info = ort_output.GetTensorTypeAndShapeInfo();
            auto out_elem_type = type_info.GetElementType();
            const std::size_t out_numel = type_info.GetElementCount();

            Tensor out;
            out.name = i < impl_->output_names_str.size() ? impl_->output_names_str[i] : "output";
            out.shape = type_info.GetShape();

            if (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                out.elem_type = TensorElementType::FLOAT32;
                const float* out_data = ort_output.GetTensorData<float>();
                out.data.assign(out_data, out_data + out_numel);
            } else if (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                out.elem_type = TensorElementType::FLOAT16;
                const Ort::Float16_t* out_data = ort_output.GetTensorData<Ort::Float16_t>();
                out.data.resize(out_numel);
                for (size_t j = 0; j < out_numel; ++j) {
                    out.data[j] = static_cast<float>(out_data[j]);
                }
            } else if (out_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                out.elem_type = TensorElementType::INT64;
                const int64_t* out_data = ort_output.GetTensorData<int64_t>();
                out.int64_data.assign(out_data, out_data + out_numel);
            } else {
                throw std::runtime_error("Unsupported model output element type");
            }
            outputs.emplace_back(std::move(out));
        }
        return outputs;
    }
    

}
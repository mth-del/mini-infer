# 修改记录

本文档用于记录后续每次修改。每条记录包含：修改原因、修改方案、结果优化。

## 2026-04-29 Qwen2.5-1.5B fp16 TensorRT engine 验证

### 1. 优化背景

- Qwen q4 ONNX 使用 `com.microsoft::MatMulNBits`，TensorRT 10.16 parser 缺少对应 plugin，无法直接转换为 engine。
- Qwen fp16 ONNX 使用外部权重格式：
  - `model_fp16.onnx` 约 1.1MB，只保存计算图和外部权重索引。
  - `model_fp16.onnx_data` 约 2.9GB，保存真实 fp16 权重。
- 当前目标是验证 Qwen fp16 ONNX 能否生成 TensorRT engine，并通过项目原生 `TensorRtBackend` 跑通一次真实 `enqueueV3()`。

### 2. 优化设计

- 使用 `curl -C -` 断点续传下载：
  - `models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_fp16.onnx`
  - `models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_fp16.onnx_data`
- 使用 `onnx_model_info` 验证 fp16 ONNX 可被正常读取：
  - 输入 59 个 tensor。
  - 输出 57 个 tensor。
  - token 输入为 `int64`，KV cache 为 `float32`。
- 先生成 decode profile engine：
  - `input_ids=[1,1]`
  - `attention_mask=[1,2]`
  - `position_ids=[1,1]`
  - `past_key_values.*=[1,2,1,128]`
- 再根据当前 `qwen_cli_cpp --prompt '你好'` 的实际 prefill 形状生成 prefill profile engine：
  - `input_ids=[1,20]`
  - `attention_mask=[1,20]`
  - `position_ids=[1,20]`
  - 初始 `past_key_values.*=[1,2,0,128]`

### 3. 优化结果

- fp16 文件大小确认：
  - `model_fp16.onnx`: `1,098,300` bytes。
  - `model_fp16.onnx_data`: `3,104,177,152` bytes。
- decode profile engine 构建通过：
  - `models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_fp16_decode_b1_s1_p1.engine`
  - engine size: `3391.6 MiB`
  - build time: `24.0842 sec`
  - deserialize time: `2.6606 sec`
- prefill profile engine 构建通过：
  - `models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_fp16_prefill_b1_s20_p0.engine`
  - engine size: `3391.9 MiB`
  - build time: `25.0616 sec`
  - deserialize time: `2.6052 sec`
- 项目原生 TensorRT 后端已跑通 Qwen fp16 prefill engine：
  - 命令：`./build/qwen_cli_cpp --provider tensorrt --model models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_fp16_prefill_b1_s20_p0.engine --prompt '你好' --max-new-tokens 1`
  - 识别 59 个 TensorRT 输入和 57 个输出。
  - `Native TensorRT backend initialized`。
  - `infer_ms=50.846`
  - `session_init_ms=4944.945`
  - `tokens_per_second=19.667`
  - 显存：session 初始化后约 `3913 MiB`。
- 当前 TensorRT 输出 token 为 `id=0`，说明执行链路已经跑通，但数值正确性还需要继续和 ONNX Runtime fp16/q4 输出对齐。

### 4. 下一步优化

- 对比 ONNX Runtime fp16 与 TensorRT fp16 的 logits argmax，确认 `id=0` 是 profile/输入问题、精度问题还是数据拷贝/类型转换问题。
- 生成同时覆盖 prefill 与 decode 的多 profile engine，避免每个阶段需要单独 engine。
- 给 `qwen_cli_cpp` 增加 TensorRT profile/engine 使用说明，明确 prompt 长度、past 长度和 attention mask 必须落在 engine optimization profile 范围内。

## 2026-04-29 原生 TensorRT engine 生成与执行验证

### 1. 优化背景

- TensorRT 10.16 SDK 已完成 CMake 接入，原生 TensorRT 分支可以编译。
- 下一步需要生成真实 `.engine` 文件，并通过项目自己的 `TensorRtBackend` 验证 engine 反序列化和 `enqueueV3()` 执行路径。
- 当前本地 Qwen q4 ONNX 模型为 `models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_q4.onnx`，输入包含 59 个动态 tensor。

### 2. 优化设计

- 先尝试使用 `trtexec` 为 Qwen q4 ONNX 构建 decode profile：
  - `input_ids=[1,1]`
  - `attention_mask=[1,2]`
  - 每层 `past_key_values.*.{key,value}=[1,2,1,128]`
- 修正 TensorRT 10 的命令行差异：
  - TensorRT 10 已不接受旧参数 `--explicitBatch`，需要移除。
- 由于 Qwen q4 ONNX 使用 `com.microsoft::MatMulNBits`，TensorRT parser 缺少对应 plugin，无法直接生成 engine。
- 为了验证项目原生 TensorRT 后端本身，新增一个最小 smoke ONNX：
  - 输入：`images`，shape `[1,3,640,640]`。
  - 输出：`output`，shape `[1,25200,85]`，匹配现有 `infer_cli` 的 YOLO 输出解析路径。
  - 使用 `trtexec` 生成 `models/trt_smoke.engine`。
- 更新 `tools/infer_cli.cpp`：
  - 增加 `--model` 作为 `-m` 别名。
  - 增加 `--provider` 作为 `-d` 别名。
  - 当 `--provider tensorrt` 且模型为 `.engine` 时，跳过 ONNX Runtime 的模型输入探测，使用默认 smoke 输入 `images:[1,3,640,640]`。

### 3. 优化结果

- Qwen q4 转 TensorRT engine 当前失败原因明确：
  - `MatMulNBits`，domain 为 `com.microsoft`。
  - TensorRT 报错：`Plugin not found, are the plugin name, version, and namespace correct?`
- smoke ONNX 成功生成 TensorRT engine：
  - `models/trt_smoke.onnx`
  - `models/trt_smoke.engine`
- 项目原生 TensorRT 后端验证通过：
  - `LD_LIBRARY_PATH=/root/autodl-tmp/tensorrt/TensorRT-10.16.1.11/lib:/root/mth/code_space/mini-infer/src/backends/onnx/lib:$LD_LIBRARY_PATH ./build/infer_cli --provider tensorrt --model models/trt_smoke.engine -l 1`
- 运行输出确认：
  - `TensorRT input name=images dtype=float32 shape=[1,3,640,640]`
  - `TensorRT output name=output dtype=float32 shape=[1,25200,85]`
  - `Native TensorRT backend initialized`
  - `Backend: tensorrt-native`
  - `Inference time(avg): 21.855 ms`
  - `Detections after NMS: 0`
- 当前测试通过：
  - `ctest --test-dir build --output-on-failure`
- `tools/infer_cli.cpp` 和 `docs/change_log_zh.md` linter 检查无新增问题。

### 4. 下一步优化

- 若要继续 Qwen TensorRT 路线，需要改用 TensorRT 支持的 fp16/fp32 ONNX，或引入支持 `MatMulNBits` 的 TensorRT plugin/转换流程。
- 为 `trt_smoke.onnx` 增加可复现生成脚本，避免只保留一次性生成产物。
- 后续可把 TensorRT engine smoke test 纳入 CTest，但需要检测本机是否存在 TensorRT SDK 和 GPU。

## 2026-04-29 TensorRT 10.16 SDK 接入验证

### 1. 优化背景

- 已在当前机器解压 TensorRT SDK：`/root/autodl-tmp/tensorrt/TensorRT-10.16.1.11`。
- SDK 目录包含 `include/NvInfer.h`、`lib/libnvinfer.so` 和 `lib/libnvonnxparser.so`，可以进入原生 TensorRT 编译路径。
- 首次启用 `MINI_INFER_TENSORRT_NATIVE=1` 编译时，TensorRT 头文件继续包含 `cuda_runtime_api.h`，但原 CMake 只链接 `cudart`，没有把 CUDA runtime include 目录加入 `mini_infer_core`。

### 2. 优化设计

- 使用以下命令重新配置 CMake：
  - `cmake -S . -B build -DMINI_INFER_ENABLE_ONNXRUNTIME=ON -DMINI_INFER_ENABLE_ORT_CUDA=ON -DMINI_INFER_ENABLE_TENSORRT=ON -DTENSORRT_ROOT=/root/autodl-tmp/tensorrt/TensorRT-10.16.1.11`
- 更新 `CMakeLists.txt`：
  - 查找 `cuda_runtime_api.h`。
  - 增加 `/usr/local/cuda-12.8/targets/x86_64-linux/include` 和对应 CUDA library hint。
  - 只有在 TensorRT include、`libnvinfer`、`cudart` 和 CUDA runtime include 都找到时，才启用 `MINI_INFER_TENSORRT_NATIVE=1`。
  - native TensorRT 分支同时加入 TensorRT include 和 CUDA runtime include。

### 3. 优化结果

- CMake 已识别原生 TensorRT：
  - `Native TensorRT backend enabled: /root/autodl-tmp/tensorrt/TensorRT-10.16.1.11/lib/libnvinfer.so`
- 原生 TensorRT 分支构建通过：
  - `cmake --build build -j2`
- 设置 TensorRT library path 后测试通过：
  - `LD_LIBRARY_PATH=/root/autodl-tmp/tensorrt/TensorRT-10.16.1.11/lib:$LD_LIBRARY_PATH ctest --test-dir build --output-on-failure`
- `qwen_cli_cpp --help` 已展示 `--provider cpu|cuda|tensorrt`。
- `CMakeLists.txt`、`src/backends/tensorrt/tensorrt_backend.cpp` 和 `tools/qwen_cli.cpp` linter 检查无新增问题。

### 4. 下一步优化

- 生成或准备真实 TensorRT `.engine`，再用 `--provider tensorrt --model <model.engine>` 验证 engine 反序列化和 `enqueueV3()`。
- 将 TensorRT SDK `lib` 路径固化为运行文档中的 `LD_LIBRARY_PATH` 步骤，或后续补充可选 RPATH 配置。
- 继续补充 Qwen ONNX 到 TensorRT engine 的 `trtexec` profile 参数。

## 2026-04-29 原生 TensorRT 后端执行路径适配

### 1. 优化背景

- 前序已经搭建原生 TensorRT 后端框架，能够完成 engine 文件读取、`IRuntime`/`ICudaEngine`/`IExecutionContext` 创建和 CUDA stream 初始化。
- `TensorRtBackend::run_many()` 仍然直接抛出 `Native TensorRT execution buffers are not implemented yet`，无法实际执行 engine。
- Qwen CLI 只能选择 `cpu` 或 `cuda` ONNX Runtime 后端，不能直接走原生 TensorRT engine。

### 2. 优化设计

- 更新 `src/backends/tensorrt/tensorrt_backend.cpp`：
  - 使用 TensorRT 8.5+ named IO tensor API 读取 engine 输入/输出名称、dtype 和 shape。
  - 初始化时打印 TensorRT engine binding inspection 信息，便于核对模型 IO。
  - 在 `run_many()` 中按 engine 输入名匹配 `Tensor`，设置动态输入 shape。
  - 增加 CUDA device buffer RAII 封装，负责输入/输出显存申请和释放。
  - 支持 `float32`、`float16`、`int32`/`int64` 输入输出转换；其中 TensorRT `int32` 输出回填为当前 `Tensor` 可表达的 `INT64`。
  - 使用 `setTensorAddress()`、`enqueueV3()`、H2D/D2H copy 和 `cudaStreamSynchronize()` 完成最小原生 TensorRT 推理路径。
- 更新 `tools/qwen_cli.cpp`：
  - `--provider` 增加 `tensorrt` 选项。
  - 当构建启用 TensorRT backend 时，`--provider tensorrt` 使用 `TensorRtBackend` 加载指定 engine。
  - 当构建未启用 TensorRT backend 时，给出明确错误提示。

### 3. 优化结果

- 当前无 TensorRT SDK 的环境仍可正常构建 stub 路径：
  - `cmake --build build -j2`
- 当前测试通过：
  - `ctest --test-dir build --output-on-failure`
- `src/backends/tensorrt/tensorrt_backend.cpp` 和 `tools/qwen_cli.cpp` linter 检查无新增问题。
- 安装 TensorRT SDK 并重新 CMake 后，可以用原生 TensorRT backend 进入真实执行路径，不再停留在初始化后抛错。

### 4. 下一步优化

- 在安装 TensorRT SDK 的环境中用真实 `.engine` 文件验证 `MINI_INFER_TENSORRT_NATIVE=1` 路径。
- 补充 ONNX 到 TensorRT engine 的生成文档，优先记录 `trtexec` 命令和 Qwen 动态 shape/profile 参数。
- 根据真实 engine 的 IO dtype 决定是否需要给 `Tensor` 增加原生 `INT32`/raw bytes 表达，减少当前 `INT64` 到 TensorRT `int32` 的临时转换。

## 2026-04-29 Qwen2.5-1.5B ONNX CUDA 部署前置调整

### 修改原因

- 当前机器是 RTX 4060 Laptop 8GB，计划在现有 `onnx_cuda` 后端部署 Qwen2.5-1.5B。
- `tools/infer_cli.cpp` 中存在一个裸露的 `在` 字符，后续重新编译会失败。
- `src/backends/onnx/ort_backend.cpp` 将 ONNX Runtime CUDA 的 `gpu_mem_limit` 写死为 2GB，不适合加载 Qwen2.5-1.5B 这类大模型。
- 当前 `infer_cli` 是 YOLO/image 推理路径，不能直接完成 Qwen 文本生成，需要明确现有能力边界。

### 修改方案

- 删除 `tools/infer_cli.cpp` 中会破坏编译的裸露字符。
- 将 ONNX Runtime CUDA 后端的显存上限从固定 2GB 改为环境变量控制：
  - 默认不设置 `gpu_mem_limit`，交给 ONNX Runtime/CUDA 自行管理。
  - 如需手动限制显存，可设置 `MINI_INFER_ORT_CUDA_GPU_MEM_LIMIT`，单位为字节。
- 新增 `docs/qwen2_5_1_5b_onnx_cuda.md`，记录当前 4060 环境下部署 Qwen2.5-1.5B 的模型选择、构建命令、CUDA smoke test 和后续需要补齐的 LLM 推理能力。

### 结果优化

- 项目重新构建通过：`cmake --build build -j2`。
- 使用 `models/yolov5n.onnx` 验证 `onnx_cuda` 后端可以正常初始化并执行。
- 避免了 2GB 显存上限导致 Qwen2.5-1.5B fp16/int4 模型无法加载的问题。
- 明确后续工作重点：扩展 `Tensor` 支持 `int64` token 输入，并新增 Qwen 专用 tokenizer/generation CLI。

## 2026-04-29 Qwen2.5-1.5B 模型下载失败处理说明

### 修改原因

- 当前环境执行 `git lfs install` 失败，提示 `git: 'lfs' 不是一个 git 命令`，说明尚未安装 Git LFS。
- 使用 `git clone https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct` 时出现 `RPC failed`、`GnuTLS recv error`、`early EOF`，属于大文件仓库下载过程中的网络/TLS 中断。
- 原部署文档只给出了 Git LFS clone 方案，缺少失败后的替代下载方式和清理步骤。

### 修改方案

- 在 `docs/qwen2_5_1_5b_onnx_cuda.md` 中补充 `git-lfs` 安装命令。
- 增加基于 `huggingface_hub.snapshot_download()` 的替代下载方案，支持断点续传，避免完全依赖 Git LFS。
- 增加失败 clone 后清理 `models/Qwen2.5-1.5B-Instruct-ONNX` 目录的命令。
- 将 CUDA smoke test 示例从损坏的 `models/yolov8n.onnx` 改为已验证可运行的 `models/yolov5n.onnx`。

### 结果优化

- 下载 Qwen2.5-1.5B ONNX 模型时有了 Git LFS 和 Hugging Face Hub 两条路径。
- 网络中断后可以通过 `resume_download=True` 继续下载，降低大模型仓库重复下载成本。
- 文档中的 CUDA 验证命令与当前本机实际可用模型一致，减少部署排查干扰。

## 2026-04-29 Qwen2.5-1.5B 最小模型下载方案

### 修改原因

- 完整下载 `onnx-community/Qwen2.5-1.5B-Instruct` 仓库约 18GB，包含 fp32、fp16、int8、q4、q4f16、bnb4 等多套模型文件。
- 当前目标是在 RTX 4060 8GB 上部署验证，不需要一次性下载所有量化版本。
- 全量下载耗时长、占用磁盘大，也更容易在网络不稳定时中断。

### 修改方案

- 将 `docs/qwen2_5_1_5b_onnx_cuda.md` 的推荐下载方式改为最小 q4 下载。
- 使用 `huggingface_hub.hf_hub_download()` 精确下载 tokenizer/config 文件和 `onnx/model_q4.onnx`。
- 将完整仓库 clone 标记为可选方案，并把目录改为 `models/Qwen2.5-1.5B-Instruct-ONNX-full`，避免和最小下载目录混淆。

### 结果优化

- 默认部署路径只下载 4060 8GB 更适合的 q4 模型，避免 18GB 全量仓库。
- 保留后续需要 fp16 或其他量化版本时的扩展空间。
- 文档中的模型目录语义更清晰：最小部署目录和完整仓库目录分开。

## 2026-04-29 Qwen2.5-1.5B q4 下载重试脚本

### 修改原因

- 使用 `hf_hub_download()` 最小下载时，网络在第 3 个文件处中断，报错 `RemoteProtocolError: Server disconnected without sending a response`。
- 当前目录中已经成功下载了 `config.json` 和 `generation_config.json`，不需要删除后从头开始。
- 原文档中的一次性 Python 片段遇到中断会直接退出，缺少逐文件重试和已下载文件跳过能力。

### 修改方案

- 新增 `tools/download_qwen_q4.py`，固定下载 Qwen2.5-1.5B q4 部署所需文件。
- 脚本会检查目标文件是否已存在，已下载文件直接跳过。
- 对每个缺失文件增加重试机制，默认最多重试 8 次，每次间隔 10 秒。
- 更新 `docs/qwen2_5_1_5b_onnx_cuda.md`，推荐使用 `python3 tools/download_qwen_q4.py` 下载，并补充 `HF_ENDPOINT=https://hf-mirror.com` 镜像重试方式。

### 结果优化

- 网络中断后可以直接重新执行脚本，不会重复下载已完成的小文件。
- 下载流程更适合不稳定网络环境，减少手动排查和重复操作。
- 保持最小 q4 下载目标不变，继续避免 18GB 全量仓库。

## 2026-04-29 Qwen2.5-1.5B q4 大文件卡住处理

### 修改原因

- `onnx/model_q4.onnx` 大文件约 1.79GB，下载时长时间停留在 `0%`，说明连接建立后没有持续传输数据。
- tokenizer/config 等小文件已经全部下载完成，当前只需要继续下载 q4 ONNX 大文件。
- 原脚本虽然支持异常重试，但对“长时间无进度”的大文件下载缺少更明确的镜像入口和超时参数。

### 修改方案

- 停止当前卡住的 `tools/download_qwen_q4.py` 下载进程。
- 更新 `tools/download_qwen_q4.py`：
  - 默认设置 `HF_HUB_DISABLE_XET=1`，减少 Xet 下载路径在不稳定网络下卡住的概率。
  - 增加 `--endpoint` 参数，可直接指定 `https://hf-mirror.com`。
  - 将元数据和下载超时默认值设置为 60 秒。
- 更新 `docs/qwen2_5_1_5b_onnx_cuda.md`，推荐使用 `python3 tools/download_qwen_q4.py --endpoint https://hf-mirror.com` 处理 Hugging Face 大文件卡住问题。

### 结果优化

- 大文件卡住后可以用镜像命令直接续下，不需要重新下载已完成的小文件。
- 下载脚本对当前网络环境更友好，减少 0% 长时间等待。
- 部署文档明确了“0% 超过 1 分钟就停止并切换镜像”的处理方式。

## 2026-04-29 Qwen2.5-1.5B q4 大文件断点续传

### 修改原因

- `onnx/model_q4.onnx` 通过 `hf_hub_download()` 下载时多次出现 `peer closed connection without sending complete message body`。
- 每次只接收到几十 KB 到几 MB 就断开，说明当前网络对 Hugging Face Hub 的大文件传输不稳定。
- 原脚本只是在 Python 下载层面重试，不能很好地保留和续传大文件的已下载部分。

### 修改方案

- 停止当前正在失败重试的下载进程。
- 更新 `tools/download_qwen_q4.py`：
  - 对 `onnx/model_q4.onnx` 增加 `curl -C -` 断点续传下载路径。
  - 增加 `--large-file-downloader` 参数，可选择 `auto`、`curl` 或 `hf`。
  - 对 q4 大文件增加最小完整大小检查，避免把未完成的部分文件误判为已下载。
  - 小文件仍使用 `huggingface_hub` 下载，保持逻辑简单。
- 更新 `docs/qwen2_5_1_5b_onnx_cuda.md`，推荐在网络不稳定时使用 `python3 tools/download_qwen_q4.py --endpoint https://hf-mirror.com --large-file-downloader curl`。

### 结果优化

- 1.79GB 的 q4 ONNX 大文件可以使用 `curl -C -` 断点续传，减少网络断开后的重复下载。
- 已完成的 tokenizer/config 小文件继续跳过，不会重复下载。
- 下载方式更适合当前慢速且容易断开的网络环境。

## 2026-04-29 GitHub 空仓库推送准备

### 修改原因

- 需要把当前代码推送到 `https://github.com/mth-del/mini-infer` 空仓库。
- 当前目录还不是 Git 仓库，需要先初始化版本控制。
- 工作区中包含 `build/`、`Testing/`、模型权重、未完成下载缓存、Nsight 报告以及 ONNX Runtime CUDA 动态库等大文件，其中 `libonnxruntime_providers_cuda.so` 超过 GitHub 普通文件大小限制，不适合直接提交。

### 修改方案

- 新增 `.gitignore`，排除构建产物、测试临时目录、模型权重、下载缓存、性能报告、Python 缓存和本地系统文件。
- 排除 `src/backends/onnx/lib/*.so*` 等 ONNX Runtime 二进制库，保留源码、头文件、脚本和文档用于推送。
- 后续通过 Git 初始化、提交并添加远端 `https://github.com/mth-del/mini-infer.git` 后推送。

### 结果优化

- 避免把数百 MB 的模型和运行时二进制文件推入 GitHub。
- 仓库内容更聚焦于源码、配置、脚本和部署说明。
- 降低首次推送失败和仓库膨胀的风险。

## 2026-04-29 Qwen2.5-1.5B ONNXRuntime LLM 链路初版

### 1. 优化背景

- Qwen q4 ONNX 不是单输入/单输出模型，不能继续沿用 YOLO smoke CLI 的单个 float Tensor 输入。
- 模型真实输入包含 `input_ids`、`attention_mask`、`position_ids` 和 28 层 `past_key_values.*`，其中 token 和 mask 都是 `int64`。
- 模型输出包含 `logits` 和 28 层 `present.*` KV cache，后续生成需要把 present 回填到下一轮 past。
- 原 `Backend::run()` 只能返回一个 `Tensor`，`Tensor` 也只能保存 `float` 数据，无法表达 Qwen 的输入输出结构。

### 2. 优化设计

- 新增 `tools/onnx_model_info.cpp`，直接通过 ONNX Runtime 打印模型 input/output 的名称、类型和 shape。
- 保留旧 `Runtime::infer()` 单输出接口，同时新增 `Runtime::infer_many()` 和 `Backend::run_many()`，给 LLM 多输出路径使用。
- 扩展 `Tensor`：
  - 增加 `TensorElementType`，支持 `float32`、`float16`、`int64`。
  - 保留 `data` 作为 float/float16 的承载，新增 `int64_data` 承载 token、mask、position id。
- 改造 `OrtBackend`：
  - 支持按 Tensor 名称绑定多个输入。
  - 支持创建 `float32`、`float16`、`int64` ORT 输入 Tensor。
  - 支持返回所有 ORT 输出，并保留输出名称、shape 和元素类型。
- 新增 `tools/qwen_cli.py` 作为端到端参考 CLI：
  - 使用 Hugging Face tokenizer 处理 chat template 和 prompt 编码。
  - 使用 ONNX Runtime 执行 greedy decode。
  - 支持从 `present.*` 输出回填 `past_key_values.*` 输入。

### 3. 优化结果

- 构建通过：
  - `cmake -S . -B build -DMINI_INFER_ENABLE_ONNXRUNTIME=ON -DMINI_INFER_ENABLE_ORT_CUDA=ON`
  - `cmake --build build -j2`
- `onnx_model_info` 已成功读取 q4 模型：
  - 输入数量：59。
  - 输出数量：57。
  - `input_ids`、`attention_mask`、`position_ids` 类型均为 `int64`，shape 均为 `[-1,-1]`。
  - 每层 KV cache 类型为 `float32`，shape 为 `[-1,2,-1,128]`。
  - `logits` 类型为 `float32`，shape 为 `[-1,-1,151936]`。
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.py --help` 可以正常输出参数说明；实际生成还需要安装 Python 依赖 `numpy`、`onnxruntime`、`transformers`。

### 4. 下一步优化

- 安装 Python 依赖后，用 `tools/qwen_cli.py` 跑 1-2 个 token 的最小生成，验证 q4 ONNX 的 KV cache 输入是否接受长度为 0 的初始 past。
- 将 `tools/qwen_cli.py` 中验证过的输入构造逻辑迁移到 C++ `qwen_cli`。
- 针对 `logits` 输出增加只取最后一个 token 的能力，避免每轮复制完整输出造成额外内存开销。
- 增加 ORT IO Binding，把 KV cache 留在 CUDA 侧，减少 CPU/GPU 来回拷贝。

## 2026-04-29 Qwen2.5-1.5B q4 Python venv 最小生成验证

### 1. 优化背景

- `tools/qwen_cli.py` 已经具备 tokenizer、ONNX Runtime Session、greedy decode 和 KV cache 回填逻辑，但本机默认 Python 环境缺少依赖。
- 为避免污染系统 Python，需要在项目内使用独立 venv 验证最小端到端生成。
- 本轮目标不是性能测试，而是先验证 q4 ONNX 能否完成 1-2 个 token 的最小生成闭环。

### 2. 优化设计

- 在项目根目录创建 `.venv`，并将 `.venv/` 加入 `.gitignore`，避免误提交虚拟环境。
- 在 venv 中安装 `numpy`、`onnxruntime`、`transformers`。
- 首次运行发现 Qwen chat template 依赖 `jinja2`，继续在 venv 中补充安装 `jinja2`。
- 使用 CPUExecutionProvider 跑最小验证命令：
  - `.venv/bin/python tools/qwen_cli.py --prompt '你好' --max-new-tokens 2 --provider cpu`

### 3. 优化结果

- venv 创建成功，pip 已升级到当前镜像源可用版本。
- 依赖安装成功：
  - `numpy`
  - `onnxruntime`
  - `transformers`
  - `jinja2`
- 最小生成验证通过：
  - 模型路径：`models/Qwen2.5-1.5B-Instruct-ONNX/onnx/model_q4.onnx`
  - Provider：`CPUExecutionProvider`
  - 输出片段：`你好！`
- 结果说明 tokenizer、ONNX Runtime 加载、空初始 KV cache、`present.*` 到 `past_key_values.*` 回填、greedy decode 主循环已经跑通。

### 4. 下一步优化

- 在 venv 中安装匹配当前 CUDA/cuDNN 环境的 `onnxruntime-gpu`，验证 `--provider cuda`。
- 增加生成耗时和 token/s 统计，区分 prefill 与 decode 阶段耗时。
- 将 Python 参考实现中已验证的输入构造逻辑迁移到 C++ `qwen_cli`。
- 后续用 ORT IO Binding 和 CUDA KV cache 驻留减少 CPU/GPU 拷贝。

## 2026-04-29 Qwen2.5-1.5B q4 算法输入输出验证

### 1. 优化背景

- CPU 最小生成已经能输出 `你好！`，但还需要确认算法层面每一步输入输出是否符合 Qwen ONNX 的真实接口。
- 重点验证 prefill 和 decode 两个阶段的 shape/dtype 变化，避免后续迁移到 C++ 时只复现表面输出而漏掉 KV cache 细节。
- 当前验证仍使用 Python 参考实现，目的是固化 C++ 版本需要照搬的输入构造规则。

### 2. 优化设计

- 使用 venv 中的 `transformers` tokenizer 构造 Qwen chat template prompt。
- 使用 `onnxruntime.InferenceSession` 加载 `onnx/model_q4.onnx`，固定走 `CPUExecutionProvider`。
- 分两步验证：
  - prefill：一次输入完整 prompt token，并给 28 层 KV cache 填长度为 0 的空张量。
  - decode：只输入上一步生成的 1 个 token，并把 `present.*` 回填为下一轮 `past_key_values.*`。
- 打印关键输入输出：
  - `input_ids`
  - `attention_mask`
  - `position_ids`
  - 第 0 层和第 27 层 KV cache
  - `logits`
  - 生成 token id 与解码文本。

### 3. 优化结果

- tokenizer 构造出的 prompt token 数为 20。
- prefill 输入：
  - `input_ids`：`int64`，shape `[1,20]`
  - `attention_mask`：`int64`，shape `[1,20]`
  - `position_ids`：`int64`，shape `[1,20]`
  - 初始 KV cache：`float32`，shape `[1,2,0,128]`
  - 总输入数：59
- prefill 输出：
  - `logits`：`float32`，shape `[1,20,151936]`
  - 第一个 greedy token id：`108386`
  - 解码文本：`你好`
  - `present.*` KV cache 长度从 0 增长到 20。
- decode 输入：
  - `input_ids`：`int64`，shape `[1,1]`
  - `attention_mask`：`int64`，shape `[1,21]`
  - `position_ids`：`int64`，shape `[1,1]`
  - KV cache：`float32`，shape `[1,2,20,128]`
  - 总输入数：59
- decode 输出：
  - `logits`：`float32`，shape `[1,1,151936]`
  - 第二个 greedy token id：`6313`
  - 解码文本：`！`
  - `present.*` KV cache 长度从 20 增长到 21。
- 两步合并生成文本为 `你好！`，说明算法输入输出闭环正确。

### 4. 下一步优化

- 将上述 prefill/decode 输入构造规则迁移到 C++ `qwen_cli`。
- C++ 迁移时必须支持长度为 0 的 KV cache 张量，以及 56 个 KV 输入/输出的名称映射。
- 增加 logits argmax 工具函数，只读取最后一个 token 的 vocab 向量。
- 后续再切换 CUDA provider，避免在 C++ 算法尚未固化前混入 CUDA 依赖问题。

## 2026-04-29 Qwen2.5-1.5B q4 CUDA Provider 验证

### 1. 优化背景

- Python 参考实现已经在 CPUExecutionProvider 上跑通，但首次使用 `--provider cuda` 时实际 fallback 到 CPU。
- 当时 ONNX Runtime 可用 provider 只有 `AzureExecutionProvider` 和 `CPUExecutionProvider`，说明 venv 中安装的是 CPU 版 `onnxruntime`。
- 需要在同一个 venv 中切换到 `onnxruntime-gpu`，确认 CUDAExecutionProvider 能被发现并实际执行最小生成。

### 2. 优化设计

- 检查当前 GPU 和驱动环境：
  - GPU：NVIDIA GeForce RTX 5090
  - Driver：580.76.05
  - 显存：32607 MiB
- 在 `.venv` 中卸载 CPU 版 `onnxruntime`，安装 GPU 版 `onnxruntime-gpu`。
- 安装后先打印 `ort.get_available_providers()`，确认 CUDA provider 是否存在。
- 使用同一个最小生成命令验证：
  - `.venv/bin/python tools/qwen_cli.py --prompt '你好' --max-new-tokens 2 --provider cuda`

### 3. 优化结果

- `onnxruntime-gpu` 安装完成，版本为 `1.25.1`。
- 当前 ONNX Runtime 可用 provider：
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`
- CUDA 最小生成验证通过：
  - 实际 providers：`['CUDAExecutionProvider', 'CPUExecutionProvider']`
  - 输出片段：`你好！`
- 这次没有出现 `CUDAExecutionProvider is not in available provider names`，说明已经不再 fallback 到纯 CPU。

### 4. 下一步优化

- 给 `tools/qwen_cli.py` 增加耗时统计，分别记录 tokenizer、session 初始化、prefill、decode 和总耗时。
- 增加显存观测命令，记录 CUDA provider 加载 q4 模型后的显存占用。
- 在 CUDA 路径验证更长输出，例如 `max_new_tokens=32`，观察 KV cache 增长后的稳定性。
- 后续将 Python 已验证的 CUDA 输入输出链路迁移到 C++ `qwen_cli`。

## 2026-04-29 Qwen2.5-1.5B q4 生成耗时统计

### 1. 优化背景

- CUDA provider 已经可以完成最小生成，但只能看到最终文本，无法判断时间主要消耗在 tokenizer、session 初始化、prefill 还是 decode。
- 后续要做 C++ 迁移、CUDA 优化和 KV cache 优化，需要先建立可重复的分阶段耗时输出。
- 当前目标是给 Python 参考 CLI 加轻量统计，不改变原有生成行为。

### 2. 优化设计

- 在 `tools/qwen_cli.py` 中使用 `time.perf_counter()` 记录阶段耗时。
- 统计项包括：
  - `tokenizer_ms`：加载 tokenizer、构造 chat template、编码 prompt 和读取 eos 配置。
  - `session_init_ms`：创建 ONNX Runtime `InferenceSession` 和读取输出元信息。
  - `prefill_ms`：第一轮完整 prompt 推理耗时。
  - `decode_ms`：后续单 token decode 累计耗时。
  - `decode_avg_ms`：后续 decode 平均单 token 耗时。
  - `generated_tokens`：本次生成 token 数。
  - `tokens_per_second`：仅按 prefill + decode 推理时间计算的生成速度。
  - `total_ms`：脚本主流程总耗时。
- 生成文本仍按原方式流式输出，结束后追加一行 `[timing]`。

### 3. 优化结果

- CUDA 2 token 最小生成验证通过：
  - 命令：`.venv/bin/python tools/qwen_cli.py --prompt '你好' --max-new-tokens 2 --provider cuda`
  - Provider：`['CUDAExecutionProvider', 'CPUExecutionProvider']`
  - 输出：`你好！`
- 本次耗时样例：
  - `tokenizer_ms=338.175`
  - `session_init_ms=8512.087`
  - `prefill_ms=301.906`
  - `decode_ms=20.417`
  - `decode_avg_ms=20.417`
  - `generated_tokens=2`
  - `tokens_per_second=6.205`
  - `total_ms=10036.454`
- `tools/qwen_cli.py` linter 检查无新增问题。

### 4. 下一步优化

- 增加 `max_new_tokens=32` 的 CUDA 稳定性和平均 decode 耗时测试。
- 增加显存统计，记录 session 初始化前后、prefill 后和 decode 后的 GPU memory。
- 优化 session 初始化耗时，后续服务化时应复用 `InferenceSession`，避免每次请求重复初始化。
- 将相同 timing 字段迁移到未来 C++ `qwen_cli`，便于 Python/C++ 对齐性能。

## 2026-04-29 Qwen2.5-1.5B q4 CUDA 32 Token 稳定性测试

### 1. 优化背景

- 2 token CUDA 最小生成只能证明 provider 和基本链路可用，无法观察 KV cache 连续增长后的稳定性。
- 需要用更长的 `max_new_tokens=32` 验证 decode 循环、`present.*` 回填和平均单 token 耗时。
- 当前阶段仍使用 Python 参考 CLI，先获取 C++ 迁移前的性能基线。

### 2. 优化设计

- 使用同一个 q4 ONNX 模型和 CUDA provider。
- 测试命令：
  - `.venv/bin/python tools/qwen_cli.py --prompt '你好，简短介绍一下你自己。' --max-new-tokens 32 --provider cuda`
- 重点观察：
  - 是否实际走 `CUDAExecutionProvider`。
  - 是否能稳定生成满 32 个 token。
  - `prefill_ms`、`decode_ms`、`decode_avg_ms` 和 `tokens_per_second`。

### 3. 优化结果

- 32 token CUDA 生成稳定完成。
- 实际 providers：`['CUDAExecutionProvider', 'CPUExecutionProvider']`。
- 输出片段：`你好！我是一个人工智能助手，可以回答问题、提供信息和帮助您完成任务。我被设计成一个助手，可以帮助您完成各种任务，`
- 本次耗时：
  - `tokenizer_ms=326.116`
  - `session_init_ms=8820.978`
  - `prefill_ms=329.339`
  - `decode_ms=292.166`
  - `decode_avg_ms=9.425`
  - `generated_tokens=32`
  - `tokens_per_second=51.488`
  - `total_ms=10661.712`
- 结果说明当前 Python CUDA 路径在 32 token decode 下稳定，KV cache 连续增长到至少 prompt 长度 + 32 token 没有触发 shape 或 provider 错误。

### 4. 下一步优化

- 增加显存统计，记录 session 初始化前后、prefill 后和 decode 后的 GPU memory。
- 增加更长输出测试，例如 `max_new_tokens=128`，观察长 decode 下的平均耗时和显存增长。
- 将当前 Python CUDA 32 token 基线作为后续 C++ `qwen_cli` 的对齐目标。

## 2026-04-29 Qwen2.5-1.5B q4 CUDA 显存统计

### 1. 优化背景

- CUDA 32 token 测试已经稳定完成，但还缺少显存占用数据。
- 后续优化 KV cache、IO Binding 和 C++ 迁移时，需要知道显存主要增长发生在 session 初始化、prefill 还是 decode 阶段。
- 当前目标是在 Python 参考 CLI 中增加轻量显存采样，作为后续性能和显存优化基线。

### 2. 优化设计

- 在 `tools/qwen_cli.py` 中新增 `gpu_memory_snapshot()`。
- 通过 `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` 采集显存。
- 记录 4 个阶段：
  - `before_session`：创建 ONNX Runtime session 前。
  - `after_session`：session 初始化和模型加载后。
  - `after_prefill`：第一轮完整 prompt 推理后。
  - `after_decode`：decode 循环结束后。
- 如果 `nvidia-smi` 不可用，则输出 `unavailable`，不影响 CPU 路径和生成流程。

### 3. 优化结果

- CUDA 2 token 生成验证通过：
  - 命令：`.venv/bin/python tools/qwen_cli.py --prompt '你好' --max-new-tokens 2 --provider cuda`
  - Provider：`['CUDAExecutionProvider', 'CPUExecutionProvider']`
  - 输出：`你好！`
- 本次耗时：
  - `tokenizer_ms=355.622`
  - `session_init_ms=11459.797`
  - `prefill_ms=304.875`
  - `decode_ms=20.362`
  - `generated_tokens=2`
  - `tokens_per_second=6.149`
  - `total_ms=13342.608`
- 本次显存：
  - `before_session`: `gpu0_used_mib=0 gpu0_total_mib=32607`
  - `after_session`: `gpu0_used_mib=4166 gpu0_total_mib=32607`
  - `after_prefill`: `gpu0_used_mib=5736 gpu0_total_mib=32607`
  - `after_decode`: `gpu0_used_mib=5736 gpu0_total_mib=32607`
- 结果说明 q4 ONNX 模型加载约占用 4.1GB 显存，prefill 后增长到约 5.7GB，2 token decode 未继续增加可见显存占用。

### 4. 下一步优化

- 用 `max_new_tokens=32` 或 `128` 复测显存统计，观察 KV cache 增长对显存的影响。
- 将显存统计和耗时统计一起作为 Python/C++ 性能对齐基线。
- 后续引入 ORT IO Binding 时，对比显存占用和 decode 平均耗时是否改善。

## 2026-04-29 Qwen2.5-1.5B q4 C++ 固定 Token 链路验证

### 1. 优化背景

- Python 参考实现已经完整验证 tokenizer、prefill、decode、KV cache 回填和 CUDA provider。
- C++ 项目下一步需要先验证 runtime/OrtBackend 本身能否按 Qwen 的 59 输入、57 输出跑通。
- 为降低变量，本轮暂不实现 C++ tokenizer，直接使用 Python 已验证的固定 prompt token ids。
- Qwen 初始 KV cache 的 shape 是 `[1,2,0,128]`，原 `OrtBackend` 会把 `data_size()==0` 判为非法输入，需要先修正。

### 2. 优化设计

- 新增 `tools/qwen_cli.cpp`，构建目标名为 `qwen_cli_cpp`。
- 固定 prompt token ids：
  - `151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198`
- C++ 输入构造对齐 Python：
  - prefill：`input_ids [1,20]`、`attention_mask [1,20]`、`position_ids [1,20]`、56 个空 KV cache `[1,2,0,128]`。
  - decode：`input_ids [1,1]`、`attention_mask [1,total_len]`、`position_ids [1,1]`、56 个上一轮 `present.*` 回填得到的 KV cache。
- 从 `logits [1,seq,151936]` 的最后一个 token vocab 向量取 argmax。
- 将 `present.N.key/value` 重命名并回填为下一轮 `past_key_values.N.key/value`。
- 修改 `OrtBackend` 输入检查逻辑：允许 `numel()==0` 的合法零长度张量。

### 3. 优化结果

- 构建通过：
  - `cmake -S . -B build -DMINI_INFER_ENABLE_ONNXRUNTIME=ON -DMINI_INFER_ENABLE_ORT_CUDA=ON`
  - `cmake --build build -j2`
- C++ CUDA 2 token 验证通过：
  - 命令：`./build/qwen_cli_cpp --provider cuda --max-new-tokens 2`
  - 第 0 步 token id：`108386`
  - 第 1 步 token id：`6313`
  - 最终输出：`[generated_ids] 108386 6313`
- 生成 token id 与 Python 参考链路完全一致，对应文本为 `你好！`。
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp`、`src/backends/onnx/ort_backend.cpp`、`CMakeLists.txt` linter 检查无新增问题。
- 当前 C++ 首版耗时明显偏高：
  - prefill：约 `55982.339 ms`
  - decode：约 `1592.424 ms`
  - 总耗时：约 `57578.364 ms`
  - 主要原因是首版会把 logits 和 56 个 KV cache 输出都拷回 CPU，后续需要减少拷贝并使用 ORT IO Binding。

### 4. 下一步优化

- 给 C++ `qwen_cli_cpp` 增加更清晰的 timing 和显存统计，对齐 Python 输出字段。
- 减少输出拷贝：优先只读取 logits 最后一个 token，同时避免每轮复制全部 KV cache 到 CPU。
- 引入 ORT IO Binding，让 KV cache 尽量驻留在 CUDA 侧。
- 在 C++ 链路稳定后，再接入 tokenizer/decoder，把 token id 输出升级为文本输出。

## 2026-04-29 Qwen2.5-1.5B q4 C++ Timing 和显存统计

### 1. 优化背景

- C++ 固定 token 链路已经能生成与 Python 一致的 token id，但上一版只打印每步 infer 耗时和总耗时。
- 为了和 Python 基线对齐，需要在 C++ `qwen_cli_cpp` 中输出 session 初始化、prefill、decode、平均 decode、token/s 和显存阶段采样。
- 后续优化输出拷贝和 ORT IO Binding 时，需要这些字段作为 C++ 侧性能对照。

### 2. 优化设计

- 在 `tools/qwen_cli.cpp` 中新增 `MsSince()` 统一毫秒计时。
- 增加 C++ 版 `GpuMemorySnapshot()`，通过 `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` 采样显存。
- 输出字段对齐 Python：
  - `session_init_ms`
  - `prefill_ms`
  - `decode_ms`
  - `decode_avg_ms`
  - `generated_tokens`
  - `tokens_per_second`
  - `total_ms`
  - `before_session`
  - `after_session`
  - `after_prefill`
  - `after_decode`
- 保持固定 token ids 和生成逻辑不变，继续用 token id 对齐 Python。

### 3. 优化结果

- 构建和 CUDA 2 token 验证通过：
  - 命令：`cmake --build build -j2 && ./build/qwen_cli_cpp --provider cuda --max-new-tokens 2`
  - 输出 token id：`108386 6313`
- 本次 C++ timing：
  - `session_init_ms=7383.730`
  - `prefill_ms=2653.488`
  - `decode_ms=37.109`
  - `decode_avg_ms=37.109`
  - `generated_tokens=2`
  - `tokens_per_second=0.743`
  - `total_ms=10322.851`
- 本次 C++ 显存：
  - `before_session`: `gpu0_used_mib=0 gpu0_total_mib=32607`
  - `after_session`: `gpu0_used_mib=4111 gpu0_total_mib=32607`
  - `after_prefill`: `gpu0_used_mib=4127 gpu0_total_mib=32607`
  - `after_decode`: `gpu0_used_mib=4127 gpu0_total_mib=32607`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp` linter 检查无新增问题。

### 4. 下一步优化

- 当前 C++ prefill 仍显著慢于 Python，优先排查输出拷贝路径和 ORT CUDA provider 配置差异。
- 减少 `OrtBackend::run_many()` 的输出复制，避免每轮把 56 个 KV cache 完整复制到 CPU。
- 引入 ORT IO Binding，让 KV cache 留在 GPU，降低 decode 阶段的内存拷贝成本。
- C++ 性能稳定后，再接 tokenizer/decoder 输出真实文本。

## 2026-04-29 Qwen2.5-1.5B q4 C++ Prefill 性能排查

### 1. 优化背景

- 上一轮 C++ CUDA 2 token 测试中，prefill 一次出现约 `2653ms`，明显慢于 Python 参考实现约 `305ms`。
- 需要优先排查 C++ 输出拷贝路径和 ONNX Runtime CUDA provider/session 配置差异，避免在错误瓶颈上过早做复杂优化。
- 本轮目标是先确认慢速是否稳定复现，并让 C++ ORT session 配置尽量对齐 Python。

### 2. 优化设计

- 对比 Python ONNX Runtime 信息：
  - Python `onnxruntime-gpu` 版本：`1.25.1`
  - 可用 provider：`TensorrtExecutionProvider`、`CUDAExecutionProvider`、`CPUExecutionProvider`
  - 默认图优化级别：`ORT_ENABLE_ALL`
  - CUDA provider option 包含 `cudnn_conv_algo_search=EXHAUSTIVE`、`arena_extend_strategy=kNextPowerOfTwo`、`do_copy_in_default_stream=1` 等。
- 对比 C++ 链接库：
  - `qwen_cli_cpp` 链接本地 `src/backends/onnx/lib/libonnxruntime.so.1`。
  - CUDA provider 已能正常启用。
- 复测 Python/C++ CUDA 2 token 耗时，判断 C++ 慢速是否稳定复现。
- 将 C++ `OrtBackend` 的 graph optimization 从 `ORT_ENABLE_EXTENDED` 调整为 `ORT_ENABLE_ALL`，对齐 Python 默认配置。

### 3. 优化结果

- Python CUDA 2 token 复测：
  - `prefill_ms=304.881`
  - `decode_ms=22.987`
  - `after_session`: `gpu0_used_mib=3729`
  - `after_prefill`: `gpu0_used_mib=3745`
- C++ 在修改前复测：
  - `prefill_ms=386.239`
  - `decode_ms=21.064`
  - `after_session`: `gpu0_used_mib=4111`
  - `after_prefill`: `gpu0_used_mib=4127`
  - 说明上一轮 `2653ms` prefill 更像冷启动或缓存波动，不是稳定瓶颈。
- C++ 改为 `ORT_ENABLE_ALL` 后：
  - `prefill_ms=316.568`
  - `decode_ms=20.531`
  - `decode_avg_ms=20.531`
  - `generated_tokens=2`
  - `tokens_per_second=5.933`
  - `total_ms=8223.941`
  - `after_session`: `gpu0_used_mib=3727`
  - `after_prefill`: `gpu0_used_mib=3743`
- C++ token id 仍与 Python 对齐：`108386 6313`。
- `ctest --test-dir build --output-on-failure` 通过。
- IDE linter 对 `onnxruntime_cxx_api.h` 的间接依赖 `arrow/result.h` 报 includePath 问题，但 CMake 实际编译通过，属于 IntelliSense includePath 配置问题。

### 4. 下一步优化

- 暂不把“输出拷贝”视为当前首要瓶颈；C++ prefill 已接近 Python 基线。
- 后续做 32 token C++ CUDA 测试，确认较长 decode 下是否仍接近 Python。
- 再进入输出复制优化和 ORT IO Binding，重点优化 KV cache 驻留 GPU 与长 decode 稳定性。
- 补齐 IDE includePath 配置，减少 ONNX Runtime 头文件的误报。

## 2026-04-29 Qwen2.5-1.5B q4 C++ CUDA 32 Token 对齐测试

### 1. 优化背景

- C++ 2 token CUDA 已经接近 Python 基线，但还需要确认较长 decode 循环下是否仍保持稳定。
- Python 32 token 基线已经完成，`decode_avg_ms` 约 `9.425ms/token`，`tokens_per_second` 约 `51.488`。
- 本轮目标是用 C++ 固定 token 链路跑满 `max_new_tokens=32`，观察长 decode 下的平均耗时和显存变化。

### 2. 优化设计

- 使用当前 `qwen_cli_cpp`，固定 prompt token ids，暂不接 tokenizer/decoder。
- 测试命令：
  - `./build/qwen_cli_cpp --provider cuda --max-new-tokens 32`
- 关注指标：
  - `prefill_ms`
  - `decode_ms`
  - `decode_avg_ms`
  - `tokens_per_second`
  - session/prefill/decode 后显存。
- 当前 C++ 测试用于压测 decode，会固定生成满 32 token；尚未实现 Python 里的 EOS 停止逻辑。

### 3. 优化结果

- C++ CUDA 32 token 生成稳定完成。
- 输出 token id 前两位仍与 Python 对齐：
  - `108386`
  - `6313`
- 本次 C++ timing：
  - `session_init_ms=7708.158`
  - `prefill_ms=301.517`
  - `decode_ms=287.022`
  - `decode_avg_ms=9.259`
  - `generated_tokens=32`
  - `tokens_per_second=54.372`
  - `total_ms=8546.531`
- 本次 C++ 显存：
  - `before_session`: `gpu0_used_mib=0 gpu0_total_mib=32607`
  - `after_session`: `gpu0_used_mib=3599 gpu0_total_mib=32607`
  - `after_prefill`: `gpu0_used_mib=3615 gpu0_total_mib=32607`
  - `after_decode`: `gpu0_used_mib=3615 gpu0_total_mib=32607`
- 与 Python 32 token 基线对比：
  - Python `prefill_ms=329.339`，C++ `prefill_ms=301.517`。
  - Python `decode_avg_ms=9.425`，C++ `decode_avg_ms=9.259`。
  - Python `tokens_per_second=51.488`，C++ `tokens_per_second=54.372`。
- 结果说明当前 C++ CUDA 较长 decode 已接近 Python 参考实现，不再存在明显 prefill/decode 性能差距。
- 注意：C++ 当前没有 EOS 停止，生成到 `151645` 后仍继续跑满 32 token；后续接文本输出前需要补齐 EOS 停止逻辑。

### 4. 下一步优化

- 给 C++ `qwen_cli_cpp` 增加 EOS token 停止逻辑，对齐 Python 行为。
- 后续再做 `max_new_tokens=128` 压测，观察更长 KV cache 下显存是否增长。
- 在接 tokenizer 前，先把 C++ 固定 token 链路的采样策略、EOS 判断和输出字段稳定下来。

## 2026-04-29 Qwen2.5-1.5B q4 C++ EOS 停止逻辑

### 1. 优化背景

- C++ 32 token 压测会固定跑满 `max_new_tokens`，即使模型已经生成 EOS token。
- Python 参考实现会在生成 `151645` 或 `151643` 时停止，并且不会把 EOS token 追加到最终生成文本。
- 为了后续接 tokenizer/decoder 后行为一致，需要先让 C++ 固定 token 链路对齐 Python 的停止策略。

### 2. 优化设计

- 在 `tools/qwen_cli.cpp` 中增加停止 token 常量：
  - `151645`
  - `151643`
- 新增 `IsStopToken()`，统一判断当前 greedy token 是否为停止 token。
- 每步 decode 后先打印 token id；如果命中 EOS：
  - 打印 `[stop] eos_token_id=...`
  - 立即退出生成循环。
  - 不把 EOS token 追加到 `generated_ids`。
- 保留原有 timing 和显存统计输出。

### 3. 优化结果

- 构建和 CUDA 32 token 验证通过：
  - 命令：`cmake --build build -j2 && ./build/qwen_cli_cpp --provider cuda --max-new-tokens 32`
- C++ 在 step 8 生成 `151645` 后停止：
  - `[stop] eos_token_id=151645`
- 最终 `generated_ids` 不包含 EOS 本身：
  - `108386 6313 104139 109944 100364 103929 101037 11319`
- 本次 timing：
  - `session_init_ms=7797.377`
  - `prefill_ms=423.977`
  - `decode_ms=115.382`
  - `decode_avg_ms=16.483`
  - `generated_tokens=8`
  - `tokens_per_second=14.832`
  - `total_ms=8632.949`
- 本次显存：
  - `after_session`: `gpu0_used_mib=3599`
  - `after_prefill`: `gpu0_used_mib=3615`
  - `after_decode`: `gpu0_used_mib=3615`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp` linter 检查无新增问题。

### 4. 下一步优化

- 后续接入 tokenizer/decoder 后，验证 C++ 文本输出是否与 Python 一致。
- 做 `max_new_tokens=128` 压测时，可增加参数控制是否忽略 EOS，以便区分真实生成和压力测试。
- 在进入 tokenizer 前，继续保持固定 token 链路作为回归测试基线。

## 2026-04-29 Qwen2.5-1.5B q4 C++ 最小 Decoder 文本输出

### 1. 优化背景

- C++ 固定 token 链路已经能输出与 Python 一致的 token id，并支持 EOS 停止。
- 下一步需要验证 C++ 文本输出是否能和 Python tokenizer decode 结果一致。
- 完整 C++ tokenizer 仍较复杂，本轮先接入最小 decoder：继续使用固定 prompt token ids，只把模型生成 token id 解码成 UTF-8 文本。

### 2. 优化设计

- 在 `tools/qwen_cli.cpp` 中新增 `ByteLevelDecoder`。
- 从 `models/Qwen2.5-1.5B-Instruct-ONNX/vocab.json` 读取 `token -> id` 映射，并反转为 `id -> token`。
- 实现最小 JSON string 解析，支持常见转义和 `\uXXXX`。
- 实现 GPT/Qwen byte-level BPE 的 byte decoder，把类似 `ä½łå¥½` 的 token string 还原为 UTF-8 文本 `你好`。
- 增加 `--vocab <vocab.json>` 参数，默认使用当前 Qwen 模型目录下的 `vocab.json`。
- 每生成一个非 EOS token：
  - 打印 `[piece] ...`
  - 将文本片段追加到最终 `[assistant] ...` 输出。

### 3. 优化结果

- 构建和 CUDA 验证通过：
  - 命令：`cmake --build build -j2 && ./build/qwen_cli_cpp --provider cuda --max-new-tokens 32`
- C++ 输出 token id：
  - `108386 6313 104139 109944 100364 103929 101037 11319`
- C++ 逐 token 文本片段：
  - `你好`
  - `！`
  - `有什么`
  - `我可以`
  - `帮助`
  - `你的`
  - `吗`
  - `？`
- C++ 最终文本输出：
  - `[assistant] 你好！有什么我可以帮助你的吗？`
- 该文本与 Python tokenizer 对相同 token ids 的 decode 结果一致。
- 本次 timing：
  - `session_init_ms=7520.924`
  - `prefill_ms=357.750`
  - `decode_ms=85.310`
  - `decode_avg_ms=12.187`
  - `generated_tokens=8`
  - `tokens_per_second=18.056`
  - `total_ms=8225.986`
- 本次显存：
  - `after_session`: `gpu0_used_mib=3853`
  - `after_prefill`: `gpu0_used_mib=3869`
  - `after_decode`: `gpu0_used_mib=3869`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp` linter 检查无新增问题。

### 4. 下一步优化

- 接入真正的 C++ tokenizer，把固定 prompt token ids 替换为从用户 prompt 动态编码。
- 完整 tokenizer 接入前，继续使用固定 token + decoder 作为 C++ 推理链路回归基线。
- 后续增加 `--ignore-eos` 参数，用于长 token 压测和真实生成两种模式切换。

## 2026-04-29 Qwen2.5-1.5B q4 C++ Tokenizer 动态 Prompt

### 1. 优化背景

- C++ 已经能用固定 prompt token ids 跑通 CUDA prefill/decode、EOS 停止和文本 decoder。
- 固定 token ids 只能作为回归基线，不能支持用户动态输入 prompt。
- 需要接入 C++ tokenizer，把 `--prompt` 文本动态编码成 Qwen chat template token ids，并验证与 Python tokenizer 对齐。

### 2. 优化设计

- 将 `ByteLevelDecoder` 扩展为 `ByteLevelTokenizer`。
- 复用当前 Qwen 模型目录文件：
  - `vocab.json`
  - `merges.txt`
- 新增 byte-level BPE encode 能力：
  - 构建 byte encoder/decoder。
  - 解析 `vocab.json` 得到 `token -> id` 和 `id -> token`。
  - 解析 `merges.txt` 得到 BPE merge rank。
  - 实现最小 pre-tokenizer，覆盖当前 Qwen chat prompt 中的英文、中文、数字、空白和标点。
  - 对普通文本执行 byte-level BPE merge 并映射到 token id。
- 新增 chat template 编码：
  - `<|im_start|>system\n...<|im_end|>\n`
  - `<|im_start|>user\n...<|im_end|>\n`
  - `<|im_start|>assistant\n`
- 新增 CLI 参数：
  - `--prompt <text>`
  - `--system <text>`
  - `--merges <merges.txt>`
- C++ 启动时打印 `[prompt_tokens]`，便于和 Python tokenizer 结果直接对比。

### 3. 优化结果

- 默认 prompt `你好` 的 C++ 动态编码与 Python 完全一致：
  - `151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 108386 151645 198 151644 77091 198`
- 中文长 prompt `你好，简短介绍一下你自己。` 的 C++ 动态编码与 Python 完全一致：
  - `151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 108386 3837 98237 99534 109432 107828 1773 151645 198 151644 77091 198`
- C++ CUDA 动态 prompt 32 token 生成通过：
  - 命令：`./build/qwen_cli_cpp --provider cuda --prompt '你好，简短介绍一下你自己。' --max-new-tokens 32`
- 输出文本：
  - `你好！我是一个人工智能助手，可以回答问题、提供信息和帮助您完成任务。我被设计成一个助手，可以帮助您完成各种任务，`
- 该输出与 Python 32 token 基线一致。
- 本次 timing：
  - `session_init_ms=7711.341`
  - `prefill_ms=438.552`
  - `decode_ms=304.730`
  - `decode_avg_ms=9.830`
  - `generated_tokens=32`
  - `tokens_per_second=43.052`
  - `total_ms=8719.891`
- 本次显存：
  - `after_session`: `gpu0_used_mib=3661`
  - `after_prefill`: `gpu0_used_mib=3677`
  - `after_decode`: `gpu0_used_mib=3677`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp` linter 检查无新增问题。

### 4. 下一步优化

- 当前 pre-tokenizer 是覆盖 Qwen 常见文本输入的最小实现，后续需要用更多英文、数字、换行、符号混合 prompt 做 tokenizer 对齐测试。
- 将 tokenizer/decoder 从 `tools/qwen_cli.cpp` 拆到独立模块，便于测试和复用。
- 增加 tokenizer 单元测试：固定 prompt 输入，对比 Python tokenizer 的 token id golden。
- 后续增加交互式 CLI 或读取 stdin，让 C++ 版可以真正作为聊天入口使用。

## 2026-04-29 Qwen2.5-1.5B q4 C++ Tokenizer 模块拆分

### 1. 优化背景

- `tools/qwen_cli.cpp` 已经包含 tokenizer、decoder、JSON 解析、byte-level BPE、推理循环、timing 和显存统计，文件膨胀到不利于维护。
- tokenizer/decoder 后续需要增加单元测试和复用，不应继续内嵌在 CLI 工具里。
- 本轮目标是保持行为不变，只做模块边界拆分。

### 2. 优化设计

- 新增头文件 `include/runtime/qwen_tokenizer.h`。
- 新增实现文件 `src/runtime/qwen_tokenizer.cpp`。
- 对外暴露 `mini_infer::QwenTokenizer`：
  - `EncodeChat(system, prompt)`
  - `Decode(token_id)`
- 使用 PIMPL 隐藏实现细节，避免在头文件暴露 vocab、merge rank、byte encoder/decoder 等内部结构。
- 将以下逻辑从 `tools/qwen_cli.cpp` 移入 `src/runtime/qwen_tokenizer.cpp`：
  - `vocab.json` 解析。
  - `merges.txt` 解析。
  - byte-level encoder/decoder。
  - 最小 JSON string parser。
  - Qwen chat template 编码。
  - BPE merge。
  - token id 到 UTF-8 文本 decode。
- 更新 `CMakeLists.txt`，将 `src/runtime/qwen_tokenizer.cpp` 加入 `mini_infer_core`。
- `tools/qwen_cli.cpp` 只保留 CLI 参数、ONNX 输入构造、推理循环、EOS、timing 和显存统计。

### 3. 优化结果

- 构建通过：
  - `cmake --build build -j2`
- 动态 prompt CUDA 验证通过：
  - 命令：`./build/qwen_cli_cpp --provider cuda --prompt '你好，简短介绍一下你自己。' --max-new-tokens 8`
- 拆分后 prompt token ids 保持不变：
  - `151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 108386 3837 98237 99534 109432 107828 1773 151645 198 151644 77091 198`
- 拆分后生成文本保持正常：
  - `你好！我是一个人工智能助手，可以`
- 本次 timing：
  - `session_init_ms=7748.443`
  - `prefill_ms=439.023`
  - `decode_ms=78.303`
  - `decode_avg_ms=11.186`
  - `generated_tokens=8`
  - `tokens_per_second=15.464`
  - `total_ms=8548.194`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp`、`src/runtime/qwen_tokenizer.cpp`、`include/runtime/qwen_tokenizer.h` 和 `CMakeLists.txt` linter 检查无新增问题。

### 4. 下一步优化

- 增加 tokenizer 单元测试，把当前两个已对齐 Python 的 prompt token ids 固化为 golden case。
- 后续将 `QwenTokenizer` 的 pre-tokenizer 覆盖范围扩展到更多英文、数字、换行和符号混合输入。
- 继续保持 `qwen_cli_cpp` 作为端到端回归入口，避免 tokenizer 优化破坏推理链路。

## 2026-04-29 Qwen2.5-1.5B q4 LLM 模型配置化

### 1. 优化背景

- `qwen_cli_cpp` 中仍硬编码了 Qwen2.5-1.5B 的层数、KV head 数、head dim、vocab size、EOS token 和 KV cache 名称。
- 这些常量会阻碍后续适配 Qwen2/Qwen2.5 同族不同尺寸模型，例如 0.5B、3B、7B。
- 当前目标是先把模型结构参数和停止 token 从代码中抽出，改为从 `config.json` / `generation_config.json` 读取。

### 2. 优化设计

- 新增 `include/runtime/llm_config.h`。
- 新增 `src/runtime/llm_config.cpp`。
- 新增 `LlmModelConfig`：
  - `num_layers`
  - `num_kv_heads`
  - `head_dim`
  - `vocab_size`
  - `bos_token_id`
  - `eos_token_id`
  - `pad_token_id`
  - `end_token_id`
  - `im_start_token_id`
  - `stop_token_ids`
  - KV cache 输入输出名称 pattern。
- 新增 `LoadLlmModelConfig(config_path, generation_config_path)`：
  - 从 `config.json` 读取 `num_hidden_layers`、`num_key_value_heads`、`hidden_size`、`num_attention_heads`、`vocab_size`、`bos_token_id`、`eos_token_id`。
  - 从 `generation_config.json` 读取 `eos_token_id` 数组和 `pad_token_id`。
  - 根据 `hidden_size / num_attention_heads` 计算 `head_dim`。
- 新增 `FormatLayerName(pattern, layer)`，用于生成 `past_key_values.N.key`、`present.N.value` 等名称。
- 更新 `qwen_cli_cpp`：
  - 新增 `--config` 和 `--generation-config` 参数。
  - `BuildInputs()` 使用配置中的层数、KV head 数和 head dim。
  - `ExtractPast()` 使用配置中的 KV 名称 pattern。
  - `ArgmaxLastToken()` 使用配置中的 vocab size。
  - EOS 判断使用配置中的 `stop_token_ids`。
- 更新 `CMakeLists.txt`，将 `src/runtime/llm_config.cpp` 加入 `mini_infer_core`。

### 3. 优化结果

- 构建通过：
  - `cmake --build build -j2`
- 当前 Qwen2.5-1.5B q4 回归验证通过：
  - 命令：`./build/qwen_cli_cpp --provider cuda --prompt '你好，简短介绍一下你自己。' --max-new-tokens 8`
- 配置化后 prompt token ids 保持不变：
  - `151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 108386 3837 98237 99534 109432 107828 1773 151645 198 151644 77091 198`
- 配置化后生成文本保持正常：
  - `你好！我是一个人工智能助手，可以`
- 本次 timing：
  - `session_init_ms=7505.026`
  - `prefill_ms=347.926`
  - `decode_ms=77.381`
  - `decode_avg_ms=11.054`
  - `generated_tokens=8`
  - `tokens_per_second=18.810`
  - `total_ms=8181.990`
- `ctest --test-dir build --output-on-failure` 通过。
- `tools/qwen_cli.cpp`、`src/runtime/llm_config.cpp`、`include/runtime/llm_config.h` 和 `CMakeLists.txt` linter 检查无新增问题。

### 4. 下一步优化

- 将 Qwen chat special token id（如 `<|im_start|>`）也从 tokenizer 配置中读取，减少 Qwen 专用硬编码。
- 增加 `LlmModelConfig` 单元测试，验证当前 Qwen config 解析出的层数、head dim、vocab size 和 stop token。
- 后续尝试替换为 Qwen2.5 0.5B/3B ONNX，验证同族模型是否只需要更换模型目录和配置文件。

## 2026-04-29 原生 TensorRT 后端框架搭建

### 1. 优化背景

- 现有 `TensorRtBackend` 只是 skeleton：`init()` 仅检查 `libnvinfer.so`，`run_many()` 直接返回输入透传，不是真实推理。
- 用户希望先搭建原生 TensorRT 后端，而不是走 ONNX Runtime TensorRT Execution Provider。
- 当前机器未安装原生 TensorRT SDK/runtime：
  - `libnvinfer` 未找到。
  - `libnvonnxparser` 未找到。
  - `trtexec` 未找到。
  - `NvInfer.h` 未找到。
- 因此本轮重点是建立可编译的原生 TensorRT 工程结构，并在缺少 SDK 时给出明确不可用提示。

### 2. 优化设计

- 更新 `CMakeLists.txt`：
  - 新增 `TENSORRT_ROOT` cache path。
  - 探测 `NvInfer.h`。
  - 探测 `libnvinfer`。
  - 探测 `libnvonnxparser`。
  - 探测 `libcudart`。
  - 如果 SDK 完整，定义 `MINI_INFER_TENSORRT_NATIVE=1` 并链接 TensorRT/CUDA runtime。
  - 如果 SDK 不完整，仍构建 TensorRT backend，但作为 unavailable stub。
- 更新 `TensorRtBackend`：
  - 增加 PIMPL，避免头文件暴露 TensorRT 类型。
  - 增加析构函数，后续负责释放 CUDA stream 和 TensorRT 资源。
  - 在 `MINI_INFER_TENSORRT_NATIVE` 路径中搭建：
    - TensorRT logger。
    - engine 文件读取。
    - `createInferRuntime()`。
    - `deserializeCudaEngine()`。
    - `createExecutionContext()`。
    - `cudaSetDevice()`。
    - `cudaStreamCreate()`。
  - 在没有 TensorRT SDK 的构建中，`init()` 明确报错并返回 false。
- 修正 `OrtBackend` 前序未完成的 provider 枚举改动，保持现有 ONNX CPU/CUDA 构建正常。

### 3. 优化结果

- 当前环境 CMake 明确提示：
  - `Native TensorRT SDK not found. TensorRT backend will build as an unavailable stub.`
- 构建通过：
  - `cmake -S . -B build -DMINI_INFER_ENABLE_ONNXRUNTIME=ON -DMINI_INFER_ENABLE_ORT_CUDA=ON -DMINI_INFER_ENABLE_TENSORRT=ON`
  - `cmake --build build -j2`
- 无 TensorRT SDK 时运行 `tensorrt` 后端会明确失败：
  - `Native TensorRT backend unavailable: libnvinfer.so not found.`
  - `Install TensorRT SDK/runtime and reconfigure CMake.`
- 不再返回假的输入透传结果，避免误判 TensorRT 推理已经可用。
- `ctest --test-dir build --output-on-failure` 通过。
- `include/runtime/tensorrt_backend.h`、`src/backends/tensorrt/tensorrt_backend.cpp`、`CMakeLists.txt`、`OrtBackend` 相关文件 linter 检查无新增问题。

### 4. 下一步优化

- 在安装 TensorRT SDK 后，验证 `MINI_INFER_TENSORRT_NATIVE=1` 路径能完成 engine 反序列化。
- 增加 TensorRT engine binding inspection，打印输入/输出 tensor 名称、dtype 和 shape。
- 实现 device buffer 分配、host/device 拷贝和 `enqueueV3()` 推理。
- 增加 ONNX 到 TensorRT engine 的构建入口，或者记录使用 `trtexec` 生成 engine 的流程。

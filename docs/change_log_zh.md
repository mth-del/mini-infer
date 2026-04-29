# 修改记录

本文档用于记录后续每次修改。每条记录包含：修改原因、修改方案、结果优化。

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

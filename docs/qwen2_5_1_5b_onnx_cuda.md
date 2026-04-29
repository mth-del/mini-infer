# Qwen2.5-1.5B ONNX CUDA Deployment

This machine has an RTX 4060 Laptop GPU with 8 GB VRAM. Qwen2.5-1.5B fits best as fp16 or int4 ONNX; avoid fp32 because weights plus KV cache and ONNX Runtime buffers can exceed available VRAM.

## Current Backend Status

- The project is built with `MINI_INFER_ENABLE_ORT_CUDA=ON`.
- ONNX Runtime CUDA provider dependencies resolve against CUDA 12 and cuDNN 9 on this machine.
- `src/backends/onnx/ort_backend.cpp` no longer hard-caps CUDA arena memory at 2 GB. To cap it manually, set `MINI_INFER_ORT_CUDA_GPU_MEM_LIMIT` to bytes before launching.
- `tools/infer_cli.cpp` is still an image/YOLO-style smoke CLI. It is useful to validate CUDA EP loading, but it is not a complete Qwen chatbot because Qwen needs tokenizer handling, `int64` token inputs, logits sampling, and a token-by-token generation loop.

## Recommended Model

Use the Hugging Face ONNX community model, but do not download the whole repository unless you need every quantization variant. The full repository is about 18 GB because it contains fp32, fp16, int8, q4, q4f16, bnb4 and tokenizer files.

For this RTX 4060 8 GB machine, start with the minimal q4 download:

```bash
python3 -m pip install --user -U huggingface_hub
python3 tools/download_qwen_q4.py
```

This keeps the download small and avoids fetching fp32/fp16/int8 variants.
The script skips files that already exist and retries each missing file if the connection is interrupted.

If Hugging Face is unstable from your network, retry with a mirror endpoint:

```bash
python3 tools/download_qwen_q4.py --endpoint https://hf-mirror.com --large-file-downloader curl
```

If the large `onnx/model_q4.onnx` file stays at `0%` for more than a minute, stop it with `Ctrl+C` and retry the mirror command above. The script skips already downloaded files and uses `curl -C -` to resume the partial q4 model file.

If you really need the full repository, install Git LFS and clone it:

```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
git clone https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct models/Qwen2.5-1.5B-Instruct-ONNX-full
```

If `git-lfs` is unavailable or `git clone` fails with `RPC failed` / `early EOF`, use the Hugging Face Hub downloader instead:

```bash
python3 -m pip install --user -U huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="onnx-community/Qwen2.5-1.5B-Instruct",
    local_dir="models/Qwen2.5-1.5B-Instruct-ONNX-full",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
```

If a failed clone leaves a partial directory, remove only that model directory before retrying:

```bash
rm -rf models/Qwen2.5-1.5B-Instruct-ONNX-full
```

If disk or VRAM is tight, prefer quantized files from the model's `onnx/` directory. For 8 GB VRAM:

- fp16 is the first choice for quality if it loads comfortably.
- int4/q4 is the fallback when fp16 leaves too little room for KV cache.

## Build

```bash
cmake -S . -B build -DMINI_INFER_ENABLE_ONNXRUNTIME=ON -DMINI_INFER_ENABLE_ORT_CUDA=ON
cmake --build build -j
```

## CUDA Smoke Test

Run an existing ONNX model first to confirm the CUDA EP can initialize:

```bash
./build/infer_cli -m models/yolov5n.onnx -d onnx_cuda -i 0 -l 3
```

Optional memory cap example:

```bash
MINI_INFER_ORT_CUDA_GPU_MEM_LIMIT=7516192768 ./build/infer_cli -m models/yolov5n.onnx -d onnx_cuda -i 0 -l 3
```

## What Is Still Needed For Qwen Text Generation

To serve Qwen through this C++ runtime, add a dedicated LLM path instead of reusing the YOLO CLI:

1. Tokenize the prompt with Qwen's tokenizer and chat template.
2. Feed ONNX inputs such as `input_ids`, `attention_mask`, `position_ids`, and optionally `past_key_values`.
3. Read the `logits` output, sample or greedy-select the next token, and append it.
4. Re-run with KV cache until `eos_token_id` or `max_new_tokens`.
5. Decode generated token ids back to UTF-8 text.

The current generic `Tensor` type only stores `float` data, so the runtime API needs an `int64` tensor representation before it can pass Qwen token ids correctly.

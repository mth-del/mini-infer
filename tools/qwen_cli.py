#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Minimal Qwen ONNX Runtime greedy decode CLI.

This script is intentionally small and explicit. It validates the model IO path
while the C++ runtime grows a native tokenizer and generation loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen ONNX with greedy decoding.")
    parser.add_argument(
        "--model-dir",
        default="models/Qwen2.5-1.5B-Instruct-ONNX",
        help="Directory containing tokenizer/config files and onnx/model_q4.onnx.",
    )
    parser.add_argument("--model", default="", help="Override ONNX model path.")
    parser.add_argument("--prompt", default="你好，介绍一下你自己。")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="ONNX Runtime execution provider.",
    )
    return parser.parse_args()


def require_deps() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependencies. Install them with:\n"
            "  python3 -m pip install --user -U numpy onnxruntime transformers\n"
            "For CUDA provider, install the onnxruntime-gpu package that matches your CUDA stack."
        ) from exc
    return np, ort, AutoTokenizer


def load_generation_ids(model_dir: Path) -> set[int]:
    config_path = model_dir / "generation_config.json"
    if not config_path.exists():
        return {151645}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    eos = config.get("eos_token_id", 151645)
    if isinstance(eos, list):
        return {int(x) for x in eos}
    return {int(eos)}


def build_prompt(tokenizer: Any, system: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def numpy_dtype(np: Any, ort_type: str) -> Any:
    if ort_type == "tensor(int64)":
        return np.int64
    if ort_type == "tensor(float16)":
        return np.float16
    if ort_type == "tensor(float)":
        return np.float32
    raise ValueError(f"Unsupported input type: {ort_type}")


def concrete_shape(shape: list[Any], past_len: int, config: dict[str, Any]) -> list[int]:
    hidden = int(config.get("hidden_size", 1536))
    heads = int(config.get("num_attention_heads", 12))
    kv_heads = int(config.get("num_key_value_heads", 2))
    head_dim = hidden // heads

    result: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            result.append(max(dim, 0))
            continue
        dim_name = str(dim).lower()
        if "batch" in dim_name:
            result.append(1)
        elif "num_key_value_heads" in dim_name:
            result.append(kv_heads)
        elif "head" in dim_name and "dim" in dim_name:
            result.append(head_dim)
        elif "past" in dim_name:
            result.append(past_len)
        else:
            result.append(1)
    return result


def present_name_for(input_name: str) -> str:
    if input_name.startswith("past_key_values"):
        return input_name.replace("past_key_values", "present", 1)
    if input_name.startswith("past."):
        return input_name.replace("past.", "present.", 1)
    return input_name


def build_inputs(
    np: Any,
    session: Any,
    token_ids: list[int],
    current_ids: list[int],
    past: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    past_len = max(0, len(token_ids) - len(current_ids))

    for meta in session.get_inputs():
        name = meta.name
        if name == "input_ids":
            inputs[name] = np.asarray([current_ids], dtype=np.int64)
        elif name == "attention_mask":
            inputs[name] = np.ones((1, len(token_ids)), dtype=np.int64)
        elif name == "position_ids":
            positions = np.arange(past_len, past_len + len(current_ids), dtype=np.int64)
            inputs[name] = positions.reshape(1, -1)
        elif name in past:
            inputs[name] = past[name]
        elif name.startswith("past"):
            shape = concrete_shape(list(meta.shape), past_len, config)
            inputs[name] = np.zeros(shape, dtype=numpy_dtype(np, meta.type))
        else:
            raise ValueError(f"Do not know how to feed model input: {name} {meta.type} {meta.shape}")

    return inputs


def gpu_memory_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "unavailable"
    values = []
    for idx, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            values.append(f"gpu{idx}_used_mib={parts[0]} gpu{idx}_total_mib={parts[1]}")
    return " ".join(values) if values else "unavailable"


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()
    np, ort, AutoTokenizer = require_deps()

    model_dir = Path(args.model_dir)
    model_path = Path(args.model) if args.model else model_dir / "onnx" / "model_q4.onnx"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt_text = build_prompt(tokenizer, args.system, args.prompt)
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    eos_ids = load_generation_ids(model_dir)
    tokenizer_ms = (time.perf_counter() - tokenizer_start) * 1000.0

    gpu_before_session = gpu_memory_snapshot()
    session_start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=providers)
    output_names = [meta.name for meta in session.get_outputs()]
    logits_name = next((name for name in output_names if "logits" in name), output_names[0])
    has_past_inputs = any(meta.name.startswith("past") for meta in session.get_inputs())
    session_ms = (time.perf_counter() - session_start) * 1000.0
    gpu_after_session = gpu_memory_snapshot()

    print(f"[model] {model_path}")
    print(f"[providers] {session.get_providers()}")
    print("[assistant]", end="", flush=True)

    past: dict[str, Any] = {}
    current_ids = list(token_ids)
    prefill_ms = 0.0
    decode_ms = 0.0
    gpu_after_prefill = "not_run"
    generated_tokens = 0
    for _ in range(args.max_new_tokens):
        feeds = build_inputs(np, session, token_ids, current_ids, past, config)
        infer_start = time.perf_counter()
        outputs = session.run(output_names, feeds)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        if generated_tokens == 0:
            prefill_ms += infer_ms
            gpu_after_prefill = gpu_memory_snapshot()
        else:
            decode_ms += infer_ms
        outputs_by_name = dict(zip(output_names, outputs))

        logits = outputs_by_name[logits_name]
        next_id = int(np.argmax(logits[0, -1, :]))
        if next_id in eos_ids:
            break

        piece = tokenizer.decode([next_id], skip_special_tokens=True)
        print(piece, end="", flush=True)
        generated_tokens += 1

        token_ids.append(next_id)
        current_ids = [next_id] if has_past_inputs else list(token_ids)

        for input_meta in session.get_inputs():
            present_name = present_name_for(input_meta.name)
            if present_name in outputs_by_name:
                past[input_meta.name] = outputs_by_name[present_name]

    print()
    gpu_after_decode = gpu_memory_snapshot()
    total_ms = (time.perf_counter() - total_start) * 1000.0
    decode_avg_ms = decode_ms / max(generated_tokens - 1, 1)
    total_gen_ms = prefill_ms + decode_ms
    tokens_per_second = generated_tokens / (total_gen_ms / 1000.0) if total_gen_ms > 0.0 else 0.0
    print(
        "[timing] "
        f"tokenizer_ms={tokenizer_ms:.3f} "
        f"session_init_ms={session_ms:.3f} "
        f"prefill_ms={prefill_ms:.3f} "
        f"decode_ms={decode_ms:.3f} "
        f"decode_avg_ms={decode_avg_ms:.3f} "
        f"generated_tokens={generated_tokens} "
        f"tokens_per_second={tokens_per_second:.3f} "
        f"total_ms={total_ms:.3f}"
    )
    print(
        "[gpu_memory] "
        f"before_session=({gpu_before_session}) "
        f"after_session=({gpu_after_session}) "
        f"after_prefill=({gpu_after_prefill}) "
        f"after_decode=({gpu_after_decode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

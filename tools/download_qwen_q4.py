#!/usr/bin/env python3
"""Download the minimal Qwen2.5-1.5B q4 ONNX files with retries."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote


DEFAULT_REPO_ID = "onnx-community/Qwen2.5-1.5B-Instruct"
DEFAULT_LOCAL_DIR = "models/Qwen2.5-1.5B-Instruct-ONNX"
Q4_MODEL_FILE = "onnx/model_q4.onnx"
Q4_MODEL_MIN_BYTES = 1_700_000_000
QWEN_Q4_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    Q4_MODEL_FILE,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download minimal Qwen2.5-1.5B q4 ONNX deployment files."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument("--etag-timeout", type=int, default=60)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument(
        "--large-file-downloader",
        choices=("auto", "curl", "hf"),
        default="auto",
        help="Downloader for onnx/model_q4.onnx. curl supports stronger resume.",
    )
    return parser.parse_args()


def is_complete(filename: str, target: Path) -> bool:
    if not target.exists() or target.stat().st_size <= 0:
        return False
    if filename == Q4_MODEL_FILE and target.stat().st_size < Q4_MODEL_MIN_BYTES:
        print(f"[resume] {filename} partial file: {target.stat().st_size} bytes")
        return False
    return True


def resolve_url(endpoint: str, repo_id: str, filename: str) -> str:
    base = (endpoint or "https://huggingface.co").rstrip("/")
    return f"{base}/{repo_id}/resolve/main/{quote(filename, safe='/')}"


def download_large_file_with_curl(args: argparse.Namespace, filename: str, target: Path) -> bool:
    curl = shutil.which("curl")
    if not curl:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_url(args.endpoint, args.repo_id, filename)
    cmd = [
        curl,
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        str(args.retries),
        "--retry-delay",
        str(int(args.sleep)),
        "--retry-all-errors",
        "--connect-timeout",
        str(args.download_timeout),
        "--output",
        str(target),
        url,
    ]
    print(f"[curl] {filename}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0 and is_complete(filename, target)


def main() -> int:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    # Xet-backed downloads can stall on some networks. Plain HTTP is easier to resume.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.etag_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.download_timeout))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing dependency. Install it with: python3 -m pip install --user -U huggingface_hub")
        return 2

    local_dir = Path(args.local_dir)

    for filename in QWEN_Q4_FILES:
        target = local_dir / filename
        if is_complete(filename, target):
            print(f"[skip] {filename}")
            continue

        if filename == Q4_MODEL_FILE and args.large_file_downloader in ("auto", "curl"):
            if download_large_file_with_curl(args, filename, target):
                continue
            if args.large_file_downloader == "curl":
                print(f"[failed] {filename}: curl download failed")
                return 1
            print(f"[fallback] {filename}: curl unavailable or failed, trying huggingface_hub")

        for attempt in range(1, args.retries + 1):
            try:
                print(f"[download] {filename} (attempt {attempt}/{args.retries})")
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    local_dir=args.local_dir,
                    etag_timeout=args.etag_timeout,
                )
                break
            except Exception as exc:  # Network errors vary between httpx/requests versions.
                if attempt == args.retries:
                    print(f"[failed] {filename}: {exc}")
                    return 1
                print(f"[retry] {filename}: {exc}")
                time.sleep(args.sleep)

    print(f"[done] Qwen q4 files are in {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download the official checkpoint with parallel Hugging Face workers."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id", default="deepseek-ai/DeepSeek-V4-Flash-0731"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output,
        max_workers=args.workers,
    )
    print(result)


if __name__ == "__main__":
    main()

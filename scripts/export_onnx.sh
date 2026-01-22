#!/usr/bin/env bash
set -euo pipefail

MODELS_PATH="${1:-/path/to/trained/models}"
STABLE_DIFF_PATH="${2:-./stable-diffusion-v1-5}"
OUTPUT_DIR="${3:-./onnx}"

python export_onnx.py \
  --unet-ckpt "${MODELS_PATH}/models/ema_ckpt.pt" \
  --stable-dif-path "${STABLE_DIFF_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cpu

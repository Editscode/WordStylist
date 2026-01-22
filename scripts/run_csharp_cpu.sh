#!/usr/bin/env bash
set -euo pipefail

ONNX_DIR="${1:-./onnx}"
OUTPUT_PATH="${2:-./outputs/wordstylist.png}"
WORD="${3:-hello}"
STYLE="${4:-0}"

dotnet run --project csharp/WordStylistCpuSample \
  --unet "${ONNX_DIR}/wordstylist_unet.onnx" \
  --vae "${ONNX_DIR}/wordstylist_vae_decoder.onnx" \
  --output "${OUTPUT_PATH}" \
  --word "${WORD}" \
  --style "${STYLE}"

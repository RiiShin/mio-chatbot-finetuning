#!/bin/bash
# Batch inference for all LoRA model variants.
# Runs 8B/14B models on 1 GPU each (parallel), 32B on 4 GPUs each (parallel).
#
# Usage:
#   bash run_infer.sh
#
# Before running, set these paths:
EVAL_JSON="data/dark_test.json"
OUTPUT_DIR="inference/infer_record"
LORA_BASE="outputs"   # where trained LoRA adapters are saved

mkdir -p "${OUTPUT_DIR}"

# --- 8B + 14B (1 GPU each, 4 jobs in parallel) ---

CUDA_VISIBLE_DEVICES=0 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_8b_1e4.json" \
    --model_name Qwen/Qwen3-8B \
    --lora_path "${LORA_BASE}/qwen3_8b_lora_1e4" \
    > "${OUTPUT_DIR}/dark_8b_1e4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_8b_5e5.json" \
    --model_name Qwen/Qwen3-8B \
    --lora_path "${LORA_BASE}/qwen3_8b_lora_5e5" \
    > "${OUTPUT_DIR}/dark_8b_5e5.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_14b_1e4.json" \
    --model_name Qwen/Qwen3-14B \
    --lora_path "${LORA_BASE}/qwen3_14b_lora_1e4" \
    > "${OUTPUT_DIR}/dark_14b_1e4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_14b_5e5.json" \
    --model_name Qwen/Qwen3-14B \
    --lora_path "${LORA_BASE}/qwen3_14b_lora_5e5" \
    > "${OUTPUT_DIR}/dark_14b_5e5.log" 2>&1 &

wait
echo "8B and 14B inference done."

# --- 32B (4 GPUs each, 2 jobs in parallel) ---

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_32b_1e4.json" \
    --model_name Qwen/Qwen3-32B \
    --lora_path "${LORA_BASE}/qwen3_32b_lora_1e4" \
    > "${OUTPUT_DIR}/dark_32b_1e4.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u inference/qwen3_infer_lora.py \
    --input_json "${EVAL_JSON}" \
    --output_json "${OUTPUT_DIR}/dark_32b_5e5.json" \
    --model_name Qwen/Qwen3-32B \
    --lora_path "${LORA_BASE}/qwen3_32b_lora_5e5" \
    > "${OUTPUT_DIR}/dark_32b_5e5.log" 2>&1 &

wait
echo "32B inference done."

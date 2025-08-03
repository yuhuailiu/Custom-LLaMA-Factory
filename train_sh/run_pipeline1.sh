#!/bin/bash
set -euo pipefail

# Record overall start time
OVERALL_START=$(date +%s)
echo "=== Working dir: $PWD ==="

# Python interpreter for merge step
PYTHON_CMD="/root/miniconda3/envs/lyh-lf/bin/python"

# Unset proxies if set
unset http_proxy https_proxy

# If no arguments are passed, fall back to hard-coded list:
if [ "$#" -ge 1 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "/data2/lyh/Custom-LLaMA-Factory/train_yaml/stage1_pretrain/qwen_7b_pretrain.yaml"
    "/data2/lyh/Custom-LLaMA-Factory/train_yaml/stage1_pretrain/qwen_7b_pretrain_with_ts.yaml"
  )
  echo "No YAMLs passed in – using default list:"
  for y in "${CONFIGS[@]}"; do
    echo "  - $y"
  done
fi

# Iterate over all chosen YAML files
for YAML_FILE in "${CONFIGS[@]}"; do
  echo
  echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Start training with: $YAML_FILE"

  # Per-job timer
  JOB_START=$(date +%s)

  # Launch training
  FORCE_TORCHRUN=1 \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    SWANLAB_MODE=disabled \
    llamafactory-cli train "$YAML_FILE"

  # Auto-merge LoRA weights
  echo ">>> Merging LoRA weights for: $YAML_FILE"
  $PYTHON_CMD auto_merge_lora.py --lora_train_yaml "$YAML_FILE"

  # Report per-job duration
  JOB_END=$(date +%s)
  JOB_DURATION=$((JOB_END - JOB_START))
  echo ">>> Finished $YAML_FILE in $JOB_DURATION seconds"
done

# Overall duration
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
echo
echo "=== All trainings completed in $TOTAL_DURATION seconds ==="

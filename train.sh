#!/bin/bash

set -euo pipefail

export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"

DATA_DIR=./data/qilin/seq_data
OUTPUT_DIR=./output
mkdir -p "$OUTPUT_DIR"

for data_file in train_rec.pkl train_src.pkl valid_rec.pkl valid_src.pkl test_rec.pkl test_src.pkl; do
    if [[ ! -f "$DATA_DIR/$data_file" ]]; then
        echo "Missing $DATA_DIR/$data_file. Generate the Qilin data first; see README.md." >&2
        exit 1
    fi
done

# 请替换为你自己的本地模型路径
BASE_MODEL=/path/to/your/t5-base-model

nohup python train.py \
    --data_path "$DATA_DIR" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --rerank \
    --filter_items \
    --test_batch_size 1 \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --epochs 50 \
    --weight_decay 0.01 \
    --galore_rank 1024 \
    --galore_update_proj_gap 200 \
    --galore_scale 1 \
    --gating_learning_rate 5e-5 \
    --initial_temperature 1.5 \
    --gating_hidden_dim 128 \
    --early_stopping_patience 5 \
    --save_steps_per_epoch 1 \
    --use_dual_space \
    --use_null_space \
    --master_port 12345 \
    > "$OUTPUT_DIR/train.txt" 2>&1 &

echo $! > "$OUTPUT_DIR/train.pid"
echo "Best-config training started in background. Process ID: $(cat "$OUTPUT_DIR/train.pid")"
echo "Log: $OUTPUT_DIR/train.txt"
echo "Training log: $OUTPUT_DIR/training.log"

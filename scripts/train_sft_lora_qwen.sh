#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 4096 \
    --dataset WestlakeNLP/DeepReview-13K \
    --input_key inputs \
    --output_key outputs \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --max_samples 36000 \
    --pretrain Qwen/Qwen3-8B \
    --save_path outputs/checkpoints/qwen-8b-review-lora \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 5 \
    --bf16 \
    --load_in_4bit \
    --attn_implementation flash_attention_2 \
    --learning_rate 5e-6 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --packing_samples \
    --aux_loss_coef 0
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

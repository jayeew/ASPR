#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset WestlakeNLP/DeepReview-13K \
   --input_key inputs \
   --output_key outputs \
   --train_batch_size 64 \
   --micro_train_batch_size 4 \
   --max_samples 36000 \
   --pretrain Qwen/Qwen3-0.6B \
   --save_path outputs/checkpoints/qwen-0.6b-review \
   --save_steps 1000 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --attn_implementation flash_attention_2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --pretrain_mode
EOF

if [[ ${1} != "slurm" ]]; then
    # 设置环境变量避免编译问题
    export DS_BUILD_FUSED_ADAM=0
    deepspeed --module $training_commands
fi

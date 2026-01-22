set fileformat=unix
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset WestlakeNLP/DeepReview-13K \
    --input_key inputs \
    --output_key outputs \
    --train_batch_size 16 \
    --micro_train_batch_size 2 \
    --max_samples 10000 \
    --pretrain Qwen/Qwen3-0.6B \
    --save_path ./checkpoint/qwen-0.6b-review-lora \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --learning_rate 5e-6 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --packing_samples \
    --load_in_4bit \
    --aux_loss_coef 0
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

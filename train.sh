#!/bin/bash
echo "====== LLM + LoRA Summarizer ======"
echo "=== Количество доступных GPU ==="
nvidia-smi -L | wc -l

echo "=== Начало обучения ==="

# Multi-GPU (6x A100)
torchrun --nproc_per_node=6 scripts/train.py \
    --config configs/qwen_lora.yaml \
    2>&1 | tee ./tmp/train_qwen.log

echo "=== Обучение завершено ==="

#!/bin/bash
echo "====== Генерация рефератов ======"

ADAPTER_PATH="checkpoints/qwen_lora/final"
CONFIG="configs/qwen_lora.yaml"

# Генерация на валидационном наборе (480 статей)
python scripts/generate.py \
    --config $CONFIG \
    --adapter $ADAPTER_PATH \
    --data dataset/dataset-480.json \
    --output results/summaries_480.json \
    2>&1 | tee ./tmp/generate_480.log

echo "=== Генерация завершена ==="
echo "Результаты: results/summaries_480.json"

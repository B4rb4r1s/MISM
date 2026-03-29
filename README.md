
Обучение (6x A100):
```bash
torchrun --nproc_per_node=6 scripts/train.py --config configs/qwen_lora.yaml
```

Override параметров на лету:
```bash
torchrun --nproc_per_node=6 scripts/train.py \
    --config configs/qwen_lora.yaml \
    --set data.max_keywords=30 training.epochs=5 training.learning_rate=1e-4
```

Генерация на 480 статьях:
```bash
python scripts/generate.py \
    --config configs/qwen_lora.yaml \
    --adapter checkpoints/qwen_lora/final \
    --data dataset/dataset-480.json \
    --output results/summaries_480.json
```

Интерактивный режим:
```bash
python scripts/generate.py \
    --config configs/qwen_lora.yaml \
    --adapter checkpoints/qwen_lora/final \
    --interactive
```

Запуск zero-shot будет таким:
```bash
python scripts/generate_zeroshot.py \
    --config configs/qwen_lora.yaml \
    --data dataset/dataset-480.json \
    --n 20 \
    --output results/zeroshot_480.json
```


Проверка установленных моделей в кэше
```bash
huggingface-cli scan-cache
```
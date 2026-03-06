Запуск тестов
```bash
pytest tests/test_phase2.py tests/test_phase3.py -v
```

Разбиение данные на выборки
```bash
python scripts/prepare_data.py --input dataset/dataset-SM-17k.json --output dataset/splits --min_summary_len 100 --val_ratio 0.10 --test_ratio 0.10 --seed 42
```

Проверка на NaN
```bash
python scripts/check_dataset.py --split dataset/splits/train.json
python scripts/check_dataset.py --split dataset/splits/test.json
python scripts/check_dataset.py --split dataset/splits/val.json
# Сразу исправить
python scripts/check_dataset.py --split dataset/splits/train.json --fix
```

Запуск обучения (Phase 4, на вашем сервере)
```bash
# 8×V100, GAZETA_2STAGE
torchrun --nproc_per_node=8 scripts/train.py --config configs/gazeta_2stage.yaml

# Возобновить с чекпоинта
torchrun --nproc_per_node=8 scripts/train.py --config configs/gazeta_2stage.yaml --resume checkpoints/gazeta_2stage/step_0005000.pt
```
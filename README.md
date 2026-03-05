Запуск тестов
```bash
pytest tests/test_phase2.py tests/test_phase3.py -v
```

Запуск обучения (Phase 4, на вашем сервере)
```bash
# 8×V100, GAZETA_2STAGE
torchrun --nproc_per_node=8 scripts/train.py --config configs/gazeta_2stage.yaml

# Возобновить с чекпоинта
torchrun --nproc_per_node=8 scripts/train.py --config configs/gazeta_2stage.yaml --resume checkpoints/gazeta_2stage/step_0005000.pt
```
echo "=== Начало обучения ==="
torchrun --nproc_per_node=8 scripts/train.py --config configs/gazeta_2stage.yaml
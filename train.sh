echo "====== MISM ======"
echo "=== Количество доступных GPU ==="
nvidia-smi -L | wc -l

echo "=== Разбиение датасета ==="
python scripts/prepare_data.py --input dataset/dataset-SM-17k.json --output dataset/splits --min_summary_len 100 --val_ratio 0.10 --test_ratio 0.10 --seed 42

echo "=== Начало обучения ==="
torchrun --nproc_per_node=6 scripts/train.py --config configs/gazeta_2stage.yaml 2>&1 | tee ./tmp/debug.log
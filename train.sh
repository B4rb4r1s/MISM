echo "====== MISM ======"
echo "=== Количество доступных GPU ==="
nvidia-smi -L | wc -l

echo "=== Разбиение датасета ==="
python scripts/prepare_data.py --input dataset/dataset-SM-17k.json --output dataset/splits --min_summary_len 100 --val_ratio 0.10 --test_ratio 0.10 --seed 42

echo "=== Начало обучения ==="
# torchrun --nproc_per_node=6 scripts/train.py --config configs/gazeta_2stage.yaml 2>&1 | tee ./tmp/debug.log
# torchrun --nproc_per_node=6 scripts/train.py --config configs/fred_2stage.yaml 2>&1 | tee ./tmp/debug.log

# Запуск с предустоновкой
torchrun --nproc_per_node=6 scripts/train.py --config configs/fred_2stage.yaml --set lambda_gen=1.0 lambda_cover=0.0 lambda_bert=0.0 lambda_gate=0.0 lambda_kw=0.0 stage2_lr=1e-5 2>&1 | tee ./tmp/debug.log

torchrun --nproc_per_node=6 scripts/train.py --config configs/fred_2stage.yaml --stage 2 --set stage2_epochs=20 stage2_lr=3e-5 lambda_gen=1.0 lambda_cover=0 lambda_bert=0 lambda_gate=0 lambda_kw=0 checkpoint_dir=checkpoints/fred_t5_1.7b_fix01

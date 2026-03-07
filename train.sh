echo "====== MISM ======"
echo "=== Количество доступных GPU ==="
nvidia-smi -L | wc -l
echo "=== Начало обучения ==="
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 scripts/train.py --config configs/gazeta_2stage.yaml
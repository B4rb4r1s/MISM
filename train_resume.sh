echo "=== Продолжение обучения ==="
torchrun --nproc_per_node=6 scripts/train.py \
    --config configs/fred_2stage.yaml \
    --stage 2 \
    --resume checkpoints/fred_t5_1.7b_fix02/best.pt \
    --set stage2_epochs=15 stage2_lr=1e-5 \
         lambda_gen=0.50 lambda_cover=0.20 lambda_bert=0.05 \
         lambda_gate=0.10 lambda_kw=0.15 \
         train_path=dataset/splits/train_clean.json \
         checkpoint_dir=checkpoints/fred_t5_1.7b_fix03

echo "=== Тесст постобработки ==="
python scripts/test_postprocess.py results/samples_val.json --save results/samples_val_pp.json -v 2>&1 | tee tmp/sample_val_pp.log
python scripts/test_postprocess.py results/samples_480.json --save results/samples_480_pp.json -v 2>&1 | tee tmp/sample_480_pp.log

echo "=== Генерация рефератов ==="
python scripts/generate_samples.py \
    --config configs/fred_2stage.yaml \
    --checkpoint checkpoints/fred_t5_1.7b_fix04/best.pt \
    --split val --n 20 \
    --num-beams 16 \
    --max-length 512 \
    --min-length 64 \
    --length-penalty 1.5 \
    --repetition-penalty 1.2 \
    --postprocess \
    --output results/samples_val_quality.json 2>&1 | tee tmp/samples_val_quality.log

python scripts/generate_samples_480.py \
    --config configs/fred_2stage.yaml \
    --checkpoint checkpoints/fred_t5_1.7b_fix04/best.pt \
    --n 20 \
    --num-beams 16 \
    --max-length 512 \
    --min-length 64 \
    --length-penalty 1.5 \
    --repetition-penalty 1.2 \
    --postprocess \
    --output results/samples_480_quality.json 2>&1 | tee tmp/samples_480_quality.log
echo "=== Тест на Val ==="
python scripts/generate_samples.py --config configs/gazeta_2stage.yaml --checkpoint checkpoints/gazeta_2stage/best.pt --split val --n 20 --num-beams 4 --min-length 64 --length-penalty 1.2
echo ""
echo "=== Тест на 480 ==="
python scripts/generate_samples_480.py --config configs/gazeta_2stage.yaml --checkpoint checkpoints/gazeta_2stage/best.pt --dataset dataset/dataset-480.json --n 20 --output results/samples_480.json --num-beams 4 --min-length 64 --length-penalty 1.2

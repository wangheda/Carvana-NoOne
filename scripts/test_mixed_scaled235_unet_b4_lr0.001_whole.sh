python test.py \
  --train_dir=model/mixed_scaled235_unet \
  --test_data_pattern=test-data/*.tfrecord \
  --batch_size=4 \
  --model=MixedScaledUNetModel \
  --run_once=True \
  --half_memory=True \
  --output_file=output/mixed_scaled235_unet_whole.csv

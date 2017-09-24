python test.py \
  --train_dir=model/mixed_scaled235_unet_train-3-accum \
  --model_checkpoint_path=model/mixed_scaled235_unet_train-3-accum/model.ckpt-34579 \
  --test_data_pattern=test-data/*.tfrecord \
  --batch_size=4 \
  --model=MixedScaledUNetModel \
  --run_once=True \
  --half_memory=True \
  --output_file=output/mixed_scaled235_unet_train-3-accum_34579.csv-part2

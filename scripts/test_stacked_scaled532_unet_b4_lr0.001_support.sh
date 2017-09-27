python test.py \
  --train_dir=model/stacked_scaled532_unet_train-1-accum_support_dataaugmentation \
  --model_checkpoint_path=model/stacked_scaled532_unet_train-1-accum_support_dataaugmentation/model.ckpt-60982 \
  --test_data_pattern=test-data/*.tfrecord \
  --batch_size=4 \
  --model=StackedScaledUNetModel \
  --run_once=True \
  --output_file=output/stacked_scaled532_unet_train-1_da.csv

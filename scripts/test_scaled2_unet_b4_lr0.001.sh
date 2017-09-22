python test.py \
  --train_dir=model/scaled3_unet_train-5-accum \
  --model_checkpoint_path=model/scaled3_unet_train-5-accum/model.ckpt-20078 \
  --test_data_pattern=test-data/*.tfrecord \
  --batch_size=4 \
  --model=ScaledUNetModel \
  --output_file=output/scaled3_unet_train-5-accum_20078.csv

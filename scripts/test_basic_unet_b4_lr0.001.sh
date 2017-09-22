python test.py \
  --train_dir=model/basic_unet_train-1 \
  --model_checkpoint_path=model/basic_unet_train-1/model.ckpt-10784 \
  --test_data_pattern=test-data/*.tfrecord \
  --batch_size=4 \
  --output_file=output/basic_unet_train-1_10784.csv

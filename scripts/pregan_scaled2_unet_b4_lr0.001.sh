python test-output-mask.py \
  --train_dir=model/scaled3_unet_train-5-accum \
  --model_checkpoint_path=model/scaled3_unet_train-5-accum/model.ckpt-20078 \
  --test_data_pattern=train-data/*.tfrecord \
  --batch_size=4 \
  --model=ScaledUNetModel \
  --run_once=True \
  --half_memory=True \
  --output_dir=train-predictions \
  --prefix=scaled2_unet_train5

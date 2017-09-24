python train.py \
  --train_dir=model/mixed_scaled235_unet \
  --train_data_pattern=train-data/*.tfrecord \
  --learning_rate_decay=0.99 \
  --num_epochs=40 \
  --accumulate_gradients=True \
  --apply_every_n_batches=2 \
  --batch_size=2 \
  --model=MixedScaledUNetModel

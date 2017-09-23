python train.py \
  --train_dir=model/mixed_scaled1235_unet_train-3-accum \
  --train_data_list=lists/train-3.list \
  --learning_rate_decay=0.97 \
  --num_epochs=20 \
  --accumulate_gradients=True \
  --apply_every_n_batches=2 \
  --batch_size=2 \
  --mixed_scaled_unet_downsample_rate="1,2,3,5" \
  --model=MixedScaledUNetModel

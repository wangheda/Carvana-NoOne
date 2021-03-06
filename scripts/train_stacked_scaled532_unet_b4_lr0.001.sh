python train.py \
  --train_dir=model/stacked_scaled532_unet_train-3-accum \
  --train_data_list=lists/train-3.list \
  --learning_rate_decay=0.97 \
  --num_epochs=20 \
  --accumulate_gradients=True \
  --apply_every_n_batches=2 \
  --batch_size=2 \
  --model=StackedScaledUNetModel

python train.py \
  --train_dir=model/basic_unet_train-4-accum \
  --train_data_list=train-4.list \
  --accumulate_gradients=True \
  --apply_every_n_batches=8 \
  --batch_size=2  \
  --base_learning_rate=0.002

python train.py \
  --train_dir=model/scaled3_unet_train-5-accum \
  --train_data_list=lists/train-5.list \
  --accumulate_gradients=True \
  --apply_every_n_batches=2 \
  --batch_size=2 \
  --model=ScaledUNetModel

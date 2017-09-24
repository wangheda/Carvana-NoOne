python train.py \
  --train_dir=model/stacked_scaled532_unet_train-3-accum_support_augmentation \
  --train_data_list=lists/train-3.list \
  --learning_rate_decay=0.99 \
  --num_epochs=60 \
  --accumulate_gradients=True \
  --apply_every_n_batches=2 \
  --batch_size=2 \
  --stacked_scaled_unet_use_support_predictions=True \
  --use_data_augmentation=True \
  --multitask=True \
  --base_learning_rate=1e-3 \
  --label_loss=MultiTaskCrossEntropyLoss \
  --support_loss_percent=0.2 \
  --support_type="label,label" \
  --model=StackedScaledUNetModel

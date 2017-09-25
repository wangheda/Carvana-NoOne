
DIR="$(pwd)"

GPU_ID=1
EVERY=2016
MODEL=StackedScaledDilatedUNetModel
MODEL_DIR="${DIR}/model/stacked_scaled532_dilated_unet_train-3-accum"

start=2016
for checkpoint in $(cd $MODEL_DIR && python ${DIR}/misc/select.py $EVERY); do
  echo $checkpoint;
  if [ $checkpoint -gt $start ]; then
    echo $checkpoint;
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --train_dir="$MODEL_DIR" \
      --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
      --eval_data_list="${DIR}/lists/test-3.list" \
      --label_loss=CrossEntropyLoss \
      --stacked_scaled_unet_use_support_predictions=True \
      --half_memory=True \
      --batch_size=4 \
      --model=$MODEL \
      --run_once=True
  fi
done


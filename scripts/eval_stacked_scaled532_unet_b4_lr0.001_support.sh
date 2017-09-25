
DIR="$(pwd)"

GPU_ID=0
EVERY=1016
MODEL=StackedScaledUNetModel
MODEL_DIR="${DIR}/model/stacked_scaled532_unet_train-3-accum_support"

start=0
for checkpoint in $(cd $MODEL_DIR && python ${DIR}/misc/select.py $EVERY); do
  echo $checkpoint;
  if [ $checkpoint -gt $start ]; then
    echo $checkpoint;
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --train_dir="$MODEL_DIR" \
      --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
      --eval_data_list="${DIR}/lists/test-3.list" \
      --label_loss=MultiTaskCrossEntropyLoss \
      --batch_size=4 \
      --model=$MODEL \
      --run_once=True
  fi
done


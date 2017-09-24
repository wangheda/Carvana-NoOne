
DIR="$(pwd)"

GPU_ID=1
EVERY=508
MODEL=MixedScaledUNetModel
MODEL_DIR="${DIR}/model/mixed_scaled235_unet_train-3-accum"

start=0
for checkpoint in $(cd $MODEL_DIR && python ${DIR}/misc/select.py $EVERY); do
  echo $checkpoint;
  if [ $checkpoint -gt $start ]; then
    echo $checkpoint;
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --train_dir="$MODEL_DIR" \
      --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
      --eval_data_list="${DIR}/lists/test-3.list" \
      --batch_size=4 \
      --model=$MODEL \
      --run_once=True
  fi
done


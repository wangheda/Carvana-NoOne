
DIR="$(pwd)"

GPU_ID=0
EVERY=20
MODEL=ScaledUNetModel
MODEL_DIR="${DIR}/model/scaled3_unet_train-5-accum"

start=6143
for checkpoint in $(cd $MODEL_DIR && python ${DIR}/misc/select.py $EVERY); do
  echo $checkpoint;
  if [ $checkpoint -gt $start ]; then
    echo $checkpoint;
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --train_dir="$MODEL_DIR" \
      --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
      --eval_data_list="${DIR}/lists/test-5.list" \
      --batch_size=4 \
      --half_memory=True \
      --model=$MODEL \
      --run_once=True
  fi
done


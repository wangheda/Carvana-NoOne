
rm train-data/*
python convert-image-into-tfrecords.py \
  --jpg_dir="train-images" \
  --gif_dir="train-masks" \
  --output_dir="train-data"

rm test-data/*
python convert-image-into-tfrecords.py \
  --jpg_dir="test-images" \
  --output_dir="test-data"

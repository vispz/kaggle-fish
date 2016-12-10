# retrain command
python ../tensorflow/tensorflow/examples/image_retraining/retrain.py \
    --image_dir=train-images/ \
    --output_graph=tensorflow-output/output_graph.pb \
    --output_labels=tensorflow-output/output_labels.txt \
    --summaries_dir=tensorflow-output/retrain_logs \
    --how_many_training_steps 10000 \
    --validation_percentage 15 \
    --testing_percentage 15 \
    --model_dir=tensorflow-output/imagenet \
    --bottleneck_dir=tensorflow-output/bottleneck \
    --flip_left_right \
    --random_crop 15 \
    --random_scale 15 \
    --random_brightness 15


# testing
python test_tf.py

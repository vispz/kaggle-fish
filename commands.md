# retrain command
python ../tensorflow/tensorflow/examples/image_retraining/retrain.py \
    --how_many_training_steps 10000 \
    --output_graph=retrained_graph.pb \ 
    --output_labels=retrained_labels.txt \ 
    --image_dir train-images/  \
    --validation_percentage 15 \
    --testing_percentage 15 \
    --flip_left_right \
    --random_crop 50 \
    --random_scale 50 \
    --random_brightness 50


# testing
python test_tf.py

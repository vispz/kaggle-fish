# Script to predict using trained a tensorflow model.
#
# Found at http://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11
import os
import re
import sys
import argparse
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import pickle
from PIL import Image
import time


def main():

    model_path = 'train_val_test_split/retrained_graph.pb'
    test_image_dir = 'test-images/'

    print("Creating Test Data from Folders in {}".format(test_image_dir))
    
    test_data=create_test_data(test_image_dir)
    print('Extracting predictions')
    predictions = predict_lbls(
        model_path=model_path,
        data=test_data
    )
    print('Dumping Test-predictions')
    pickle.dump(predictions, open('test-predictions-retrained', 'wb'))
    print('Pandas')
    test_df = pd.DataFrame(
        data=predictions,
        # This ordering is taken from the file given to
        # `--output_labels` in `retrain.py`.
        columns=[
            'SHARK',
            'DOL',
            'NOF',
            'LAG',
            'ALB',
            'YFT',
            'OTHER',
            'BET',
        ],
    )
    test_df.loc[:,'image'] = [
        os.path.split(img_data['img'])[1] #Get the tail portion file.end
        for img_data in test_data
    ]
    test_df.to_csv(
        path_or_buf='submission.csv',
        sep=',',
        index=False
    )
    print("Bye")


def create_test_data(test_dir):
    return [
        {
            "img":os.path.join(test_dir,img)
            #Leave out 'lbl' since this is test
        } 
        for img in os.listdir(test_dir) if re.search('jpg|JPG', img)
    ]

def predict_lbls(model_path, data):
    create_graph(model_path=model_path)
    predictions = []
    with tf.Session() as sess:
        final_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for ind, img_info in enumerate(data):
            if (ind%10 == 0) and ind:
                print('Processing {img} {lbl}...'.format(
                    img=img_info['img'],
                    lbl=img_info.get('lbl', 'Test')
                ))
            if not gfile.Exists(img_info['img']):
                tf.logging.fatal('File does not exist %s', img_info['img'])

            image_data = gfile.FastGFile(img_info['img'], 'rb').read()
            predictions.append(
                sess.run(
                    final_tensor,
                    {'DecodeJpeg/contents:0': image_data}
                ).ravel().tolist(),
            )
    return predictions


def create_graph(model_path):
    with gfile.FastGFile(model_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


if __name__ == "__main__":
    main()
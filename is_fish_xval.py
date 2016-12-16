#found at http://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11
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
from matplotlib import pyplot as plt
import matplotlib
import pickle
from PIL import Image
import time
import xgboost
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import fbeta_score, make_scorer
import random

random.seed(10)


#1)understand what inception layers mean and # of features as each layers
#2)Use Gausien kernnal(SVM with kernal) instead of LinearSVC.
#3)GridSearchRandomSearchCV for parameters to SVM (using cross validation to be confident on metrics)(linear hinge loss or log loss)
#4)When submission train on all

def main():

    plt.figure(figsize=(17,10))
    model_dir = 'imagenet'
    images_dir = 'train-images/'
    fish_folder_map = {
        "Albacore": "ALB",
        "Bigeye": "BET",
        "Mahi Mahi": "DOL",
        "Opah": "LAG",
        "Sharks": "SHARK",
        "Yellow Fin": "YFT",
        "Other": "OTHER",
        "NoFish": "NoF",
    }
    test_image_dir = 'test-images/'
    print("Creating is Fish Train Data")
    is_fish_train_data= create_train_data(images_dir, fish_folder_map, [
        "Albacore", "Bigeye", "Mahi Mahi", "Opah", "Sharks", "Yellow Fin", "Other"])
    print("Creating Train Data from Folders in {}".format(images_dir))
    train_data= create_train_data(images_dir, fish_folder_map)

    if not os.path.isfile('features') or not os.path.isfile('labels'):
        print('Extracting Features')
        features,labels = extract_features(
            model_dir=model_dir,
            data=train_data
        )
        print('Dumping Features and Labels')
        pickle.dump(features, open('features', 'wb'))
        pickle.dump(labels, open('labels', 'wb'))
    else:
        print('Found features and labels. Loading...')
        features = pickle.load(open('features'))
        labels = pickle.load(open('labels'))

    if not os.path.isfile('features-is-fish') or not os.path.isfile('labels-is-fish'):
        print('Extracting Is Fish Features')
        features_is_fish,labels_is_fish = extract_features(
            model_dir=model_dir,
            data=is_fish_train_data
        )
        print('Dumping Is Fish Features and Labels')
        pickle.dump(features_is_fish, open('features-is-fish', 'wb'))
        pickle.dump(labels_is_fish, open('labels-is-fish', 'wb'))
    else:
        print('Found Is Fish features and labels. Loading...')
        features_is_fish = pickle.load(open('features-is-fish'))
        labels_is_fish = pickle.load(open('labels-is-fish'))

    X_train= features
    y_train= labels
    X_is_fish_train= features_is_fish
    y_is_fish_train= map(lambda x: int(x!='FISH'), labels_is_fish)
    clf_is_fish = svm.SVC(
        kernel='poly', 
        probability=True, 
        random_state=324

    )
    search_clf= model_selection.GridSearchCV(
        estimator=clf_is_fish,
        param_grid={
                'C':[0.001],
                'degree': [2],
            },
        n_jobs=3,
        cv=3,
        refit=True,
        verbose=3,
        scoring=make_scorer(f1_score),
    )
    search_clf.fit(X_is_fish_train, y_is_fish_train)
    print('Best score of: {score} params:\n{params}'.format(
        score=search_clf.best_estimator_,
        params=search_clf.best_params_
    ))
    print("Bye")




def create_train_data(images_dir, fish_folder_map, fish_types=None):
    train_data= [
        {
            "img":os.path.join(images_dir,folder,img),
            "lbl": label if not fish_types else "FISH" if label in fish_types else "NoF"
        } 
        for label, folder in fish_folder_map.iteritems()
        for img in os.listdir(os.path.join(images_dir,folder)) if re.search('jpg|JPG', img)
    ]
    return train_data


def create_test_data(test_dir):
    return [
        {
            "img":os.path.join(test_dir,img)
            #Leave out 'lbl' since this is test
        } 
        for img in os.listdir(test_dir) if re.search('jpg|JPG', img)
    ]

def extract_features(model_dir, data):
    nb_features = 2048
    features = np.empty((len(data),nb_features))
    labels = []
    create_graph(model_dir=model_dir)
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, img_info in enumerate(data):
            if (ind%10 == 0):
                print('Processing {img} {lbl}...'.format(
                    img=img_info['img'],
                    lbl=img_info.get('lbl', 'Test')
                ))
            if not gfile.Exists(img_info['img']):
                tf.logging.fatal('File does not exist %s', img_info['img'])

            image_data = gfile.FastGFile(img_info['img'], 'rb').read()
            predictions = sess.run(
                next_to_last_tensor,
                {'DecodeJpeg/contents:0': image_data}
            )
            features[ind,:] = np.squeeze(predictions)
            if img_info.get('lbl', False):
                labels.append(img_info['lbl'])
    return features, labels

def create_graph(model_dir):
    with gfile.FastGFile(
            os.path.join(model_dir, 'classify_image_graph_def.pb'),
            'rb'
        ) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def plot_confusion_matrix(y_true,y_pred, show=True):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:,:], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    if show:
        plt.show()
    else:
        plt.draw()


def plot_label_freq(label_counts_map, show=True):
    objects = [l for l in label_counts_map]
    y_pos = np.arange(len(objects))
    performance = [label_counts_map[o] for o in objects]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Count')
    plt.title('Frequency of Training Data')
    if show:
        plt.show()
    else:
        plt.draw()

if __name__ == "__main__":
    main()
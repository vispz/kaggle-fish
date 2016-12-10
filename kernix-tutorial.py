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


#1)understand what inception layers mean and # of features as each layers
#2)Use Gausien kernnal(SVM with kernal) instead of LinearSVC.
#3)GridSearchRandomSearchCV for parameters to SVM (using cross validation to be confident on metrics)(linear hinge loss or log loss)
#4)When submission train on all

def main():
    cmd_opts = _parse_args()

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
    print("Creating Train Data from Folders in {}".format(images_dir))
    train_data=create_train_data(images_dir, fish_folder_map, cmd_opts.extra_train_dir)
    train_lbl_counts={
        lbl: sum([img_info['lbl']==lbl for img_info in train_data])
        for lbl in fish_folder_map
    }
    if not cmd_opts.produce_sub:
        plt.subplot(121)
        plot_label_freq(train_lbl_counts, show=False)

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
    if cmd_opts.valid_size > 0 and not cmd_opts.produce_sub:
        print('Splitting data into train validation, of valid_size: {}'.format(
            cmd_opts.valid_size
        ))
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
            features, 
            labels, 
            test_size=cmd_opts.valid_size, 
            random_state=42,
            stratify=labels #XXX(try not strat)
        )
    else:
        X_train = features
        y_train = labels
    if not cmd_opts.search_cv:
        if False:
            clf = svm.SVC(
                kernel='poly', 
                C=0.001, 
                gamma=0.1, 
                probability=True, 
                random_state=324
            )
        else:
            print ("Using XGBoost")
            clf= xgboost.XGBClassifier()
        print("Fitting Last Layer")
        clf.fit(X_train, y_train)
        if cmd_opts.valid_size > 0 and not cmd_opts.produce_sub:
            print("Predicting on Validation")
            y_pred = clf.predict(X_valid)
            print("Accuracy: {0:0.1f}%".format(accuracy_score(y_valid,y_pred)*100))
        else:
            print("Skipping Validation since size is 0")
    else:
        if False:
            model_params= {
                #'kernel':['rbf', 'poly'],
                #'kernel':['rbf', 'poly'],
                'C':[0.001,0.01,0.1,1,10],
                'degree': [2,3,5,7],
                #'gamma':[0.001,0.01,0.1,1,10],
                'probability':[True],
                'random_state':[324],
            }
            model= svm.SVC()
        else:
            model_params= {
                #'kernel':['rbf', 'poly'],
                #'kernel':['rbf', 'poly'],
                'C':[0.001,0.01,0.1,1,10],
                'degree': [2,3,5,7],
                #'gamma':[0.001,0.01,0.1,1,10],
                'probability':[True],
                'random_state':[324],
            }
            model = xgboost.XGBClassifier()
        search_clf= model_selection.GridSearchCV(
            estimator=model, 
            param_grid=model_params,
            n_jobs=7,
            cv=5,
            refit=True,
            verbose=3,
        )
        print('Starting Grid Search')
        search_clf.fit(X_train, y_train)
        print('Best score of: {score} params:\n{params}'.format(
            score=search_clf.best_estimator_,
            params=search_clf.best_params_
        ))
        clf=search_clf.best_estimator_
    if not cmd_opts.produce_sub:
        plt.subplot(122)
        plot_confusion_matrix(y_true=y_valid,y_pred=y_pred, show=True)
    else:
        print("Creating Test Data from Folders in {}".format(test_image_dir))
        test_data=create_test_data(test_image_dir)
        if not os.path.isfile('test-features'):
            print('Extracting Features')
            test_features,_ = extract_features(
                model_dir=model_dir,
                data=test_data
            )
            print('Dumping Test-Features')
            pickle.dump(test_features, open('test-features', 'wb'))
        else:
            print('Found test features. Loading...')
            test_features = pickle.load(open('test-features'))
        print('Predicting Test')
        test_pred = clf.predict(test_features)
        csv_headers = [
            fish_folder_map[c]
            for c in clf.classes_
        ]
        #Do one hot encoding or get predicted probabilities
        if not cmd_opts.one_hot:
            print('Predicting Probabilities')
            test_proba = clf.predict_proba(test_features)
        else:
            print('Creating One-Hot encoding')
            test_proba=np.zeros((len(test_pred), len(clf.classes_)))
            for row in xrange(len(test_pred)):
                test_proba[row][csv_headers.index(fish_folder_map[test_pred[row]])]=1

        test_df = pd.DataFrame(
            data=test_proba,
            columns=csv_headers
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
        badCount=0
        for i in xrange(len(test_pred)):
            maxI=np.where(test_proba[i]==max(test_proba[i]))[0][0]
            if test_pred[i] != clf.classes_[maxI]:
                badCount+=1
        print("Bad count of: {}".format(badCount))


        #for i in xrange(len(test_pred)):
        #    print(test_pred[i])
        #    image = Image.open(test_data[i]['img'])
        #    image.show()
        #    time.sleep(2)
        #    image.close()


    print("Bye")


def _parse_args():
    """Parse command line arguments"""
    args = sys.argv[1:]
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        '--produce-sub',
        dest='produce_sub',
        help='Produce submision file',
        default=False,
        action='store_true',
    )
    cmd_parser.add_argument(
        '--one-hot',
        dest='one_hot',
        help='Use one hot encoding for submision',
        default=False,
        action='store_true',
    )
    cmd_parser.add_argument(
        '--valid-size',
        dest='valid_size',
        help='The size to make the holdout validation set',
        type=float,
        default=0.2,
    )
    cmd_parser.add_argument(
        '--extra-train-dir',
        dest='extra_train_dir',
        help='Dir of extra train images',
        type=str,
    )
    cmd_parser.add_argument(
        '--search-cv',
        dest='search_cv',
        help='Perform Search of parameters',
        default=False,
        action='store_true',
    )
    cmd_opts = cmd_parser.parse_args(args=args)
    return cmd_opts


def create_train_data(images_dir, fish_folder_map, extra_train_dir=None):
    train_data= [
        {
            "img":os.path.join(images_dir,folder,img),
            "lbl":label
        } 
        for label, folder in fish_folder_map.iteritems()
        for img in os.listdir(os.path.join(images_dir,folder)) if re.search('jpg|JPG', img)
    ]

    if extra_train_dir:
        extra_train_dir=[
            {
                "img":os.path.join(extra_train_dir,folder,img),
                "lbl":label
            } 
            for label, folder in fish_folder_map.iteritems() if os.path.isdir(os.path.join(extra_train_dir,folder))
            for img in os.listdir(os.path.join(extra_train_dir,folder)) if re.search('jpg|JPG', img)
        ]
        train_data.extend(extra_train_dir)

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
"""This script splits the train images into train and valid for
tensorflow.

.. code-block:: bash

    python split_train_valid.py \
        --input-path train-images \
        --train-path split_data/train \
        --validation-path split_data/valid \
        --train-ratio 0.7 \
        --silent
"""
import argparse
import logging
import os
import os.path
import random
import shutil


logging.basicConfig(level=logging.DEBUG)
random.seed(1)


def main():
    cmd_opts = _parse_args()
    _setup_dest_folders(
        train_path=cmd_opts.train_path,
        valid_path=cmd_opts.valid_path,
        is_silent=cmd_opts.is_silent,
    )
    for each_lbl in os.listdir(cmd_opts.input_path):
        split_and_save(
            input_path_1lbl=os.path.join(cmd_opts.input_path, each_lbl),
            train_path_1lbl=os.path.join(cmd_opts.train_path, each_lbl),
            valid_path_1lbl=os.path.join(cmd_opts.valid_path, each_lbl),
            train_ratio=cmd_opts.train_ratio,
        )


def _parse_args():
    """Parse command line arguments"""
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        '-ip',
        '--input-path',
        dest='input_path',
        help='Path to the input train images',
        required=True,
        type=str,
    )
    cmd_parser.add_argument(
        '-tp',
        '--train-path',
        dest='train_path',
        help='Path to the split train images',
        required=True,
        type=str,
    )
    cmd_parser.add_argument(
        '-vp',
        '--validation-path',
        dest='valid_path',
        help='Path to the validation images',
        required=True,
        type=str,
    )
    cmd_parser.add_argument(
        '-r',
        '--train-ratio',
        dest='train_ratio',
        help='The ratio of train images',
        required=True,
        type=float,
    )
    cmd_parser.add_argument(
        '-s',
        '--silent',
        dest='is_silent',
        help='Silently delete the folder output contents if exists',
        default=False,
        action='store_true',
    )
    cmd_opts = cmd_parser.parse_args()
    return cmd_opts


def _setup_dest_folders(train_path, valid_path, is_silent):
    """
    is_silent: bool: if true we delete folder even if not empty. if
        false and folder not empty, we crash.
    """
    logging.info('Deling and creating train folder %s', train_path)
    _del_and_create_dir(path=train_path, is_silent=is_silent)
    logging.info('Deling and creating validation folder %s', valid_path)
    _del_and_create_dir(path=valid_path, is_silent=is_silent)


def _del_and_create_dir(path, is_silent):
    _del_dir(path=path, is_silent=is_silent)
    os.makedirs(path)


def _del_dir(path, is_silent):
    """Deletes a folder.

    is_silent: bool: if true we delete folder even if not empty. if
        false and folder not empty, we crash.
    """
    delete_fun = shutil.rmtree if is_silent else os.rmdir
    if os.path.exists(path):
        delete_fun(path)


def split_and_save(
        input_path_1lbl, train_path_1lbl, valid_path_1lbl,
        train_ratio):
    logging.debug(
        'Coping `{inp}` to `{trn}` and `{vld}`.'.format(
            inp=input_path_1lbl,
            trn=train_path_1lbl,
            vld=valid_path_1lbl,
        ),
    )
    img_fl_names = [
        f for f in os.listdir(input_path_1lbl)
        if (
            os.path.isfile(os.path.join(input_path_1lbl, f)) and
            (f.endswith('JPG') or f.endswith('jpg'))
        )
    ]
    random.shuffle(img_fl_names)
    num_train = int(train_ratio * len(img_fl_names))
    for i, each_img_fl in enumerate(img_fl_names):
        dst_fldr = train_path_1lbl if i < num_train else valid_path_1lbl
        _create_dir_silently(path=dst_fldr)
        src = os.path.join(input_path_1lbl, each_img_fl)
        dst = os.path.join(dst_fldr, each_img_fl)
        #logging.debug('Coping {src} to {dst}'.format(src=src, dst=dst))
        shutil.copyfile(src=src, dst=dst)


def _create_dir_silently(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()

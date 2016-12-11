"""Script to compare mxe diff between submissions.

Sample Usage::

python mxe_diff.py \
    --old-sub kaggle-submissions/submission-v1.csv \
    --new-sub kaggle-submissions/submission-v10-Poly-degree2-C10.csv \
    --num-top-mismatch 5 \
    --no-plot-mismatch
"""
import argparse
import pandas as pd
import numpy as np
import os.path
import pprint
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg


TEST_IMGS_DIR = 'test-images'


def main():
    cmd_opts = _parse_args()
    df_and_mxes = load_and_compute_mxe(
        old_sub=cmd_opts.old_sub,
        new_sub=cmd_opts.new_sub,
    )
    print 'The mxe:', df_and_mxes['mxes']['mxe']
    if cmd_opts.num_top_mismatch:
        top_mismatch_fls = get_top_mismatch(
            mxe_list=df_and_mxes['mxes']['mxe_list'],
            img_names=df_and_mxes['old_sub_df'].image.tolist(),
            n_top=cmd_opts.num_top_mismatch,
        )
        print 'Top mismatched files: '
        pprint.pprint(top_mismatch_fls)
        if cmd_opts.to_plot_mismatch:
            plot_top_mimatch(
                top_mismatch_fls=top_mismatch_fls,
                test_imgs_dir=TEST_IMGS_DIR,
            )


def _parse_args():
    """Parse command line arguments"""
    cmd_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cmd_parser.add_argument(
        '-o',
        '--old-sub',
        dest='old_sub',
        type=str,
        help='The old submission file (this is assumed as the true labels).',
        required=True,
    )
    cmd_parser.add_argument(
        '-n',
        '--new-sub',
        dest='new_sub',
        type=str,
        help='The new submission file.',
        required=True,
    )
    cmd_parser.add_argument(
        '-m',
        '--num-top-mismatch',
        dest='num_top_mismatch',
        type=int,
        help=(
            'Number of top mismatched image file names to print out and plot'
        ),
        default=5,
    )
    cmd_parser.add_argument(
        '-np',
        '--no-plot-mismatch',
        dest='to_plot_mismatch',
        help=(
            'If set mismatches will not be plotted.'
        ),
        action='store_false',
        default=True,
    )
    return cmd_parser.parse_args()


def load_and_compute_mxe(old_sub, new_sub):
    """
    old_sub: Old submission file path.
    new_sub: New submission file path.
    """
    old_sub_df = load_sub(fl=old_sub)
    new_sub_df = load_sub(fl=new_sub)
    # Assert that the two files have the same images (in order)
    assert old_sub_df.image.tolist() == new_sub_df.image.tolist()

    lbls = sorted(c for c in new_sub_df.columns if c != 'image')
    # Assume that the old file is the true values and the new values
    #   are the predictions
    mxes = compute_mxe(y=old_sub_df[lbls].values, p=new_sub_df[lbls].values)
    return {
        'old_sub_df': old_sub_df,
        'new_sub_df': new_sub_df,
        'mxes': mxes,
    }


def load_sub(fl):
    df = pd.read_csv(fl)
    return df.sort_values(by='image')


def compute_mxe(y, p):
    mxe_list = -(y*np.log(p)).sum(axis=1)
    return {
        'mxe_list': mxe_list.tolist(),
        'mxe': mxe_list.mean(),
    }

def get_top_mismatch(mxe_list, img_names, n_top):
    top_mismatch_inds = top_inds(
        arr=np.array(mxe_list),
        n=n_top,
    )
    return (np.array(img_names)[top_mismatch_inds]).tolist()


def top_inds(arr, n):
    """Indices of top n values in an array.

    Source: http://stackoverflow.com/a/6910672
    """
    return np.array(arr).argsort()[-n:][::-1]


def plot_top_mimatch(top_mismatch_fls, test_imgs_dir):
    for fl in top_mismatch_fls:
        plt.figure()
        image = mpimg.imread(os.path.join(test_imgs_dir, fl))
        plt.imshow(image)
        plt.show(block=False)
    try:
        input('Type any key to exit...')
    except SyntaxError:
        pass


if __name__ == '__main__':
    main()
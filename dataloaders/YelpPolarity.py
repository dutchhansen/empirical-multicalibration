from dataloaders.utils.download_utils import download_dataset
from utils import href
import numpy as np
import pandas as pd
import os

def groups_map(features_df, groups='default'):
    # create groups based on containment
    keywords = [
        # pricing
        'pricey',
        'expensive',
        'cheap',
        'affordable',
        'overpriced',
        # business
        'service',
        'ambiance',
        'time',
        'location',
        'late',
        'early',
        'loud',
        'quiet'
        'music',
        # cuisine
        'breakfast',
        'lunch',
        'dinner',
        'sandwich',
        'pizza',
        'salad',
        'fish',
        'grill',
        'pasta',
        'burger',
        'sushi',
        'drinks',
        'coffee',
        # quality
        'salty',
        'sweet',
        'large',
        'small',
        # judgement
        'hype',
        'garbage',
        'terrible',
        'incredible',
        'love',
        'again',
    ]
    groups_map = {
        keyword: np.where(features_df['text'].str.contains(keyword, regex=False))[0]
        for keyword in keywords
    }

    return groups_map


def load_YelpPolarity(groups='default'):
    """
    Yelp Polarity dataset. Dataset documents reviews of businesses
    made on Yelp and associated user metadata.
    This is a variant of the 2015 Yelp Dataset Challenge dataset.

    Input (x): Yelp review text.
    Label (y): 1 if review >= 3/4 stars, 0 if review < 3/4 stars.

    Website:
        Dataset available at https://course.fast.ai/datasets.
        Also in the README of https://github.com/zhangxiangxiao/Crepe.

    Original publication:
        @article{zhangCharacterlevelConvolutionalNetworks2015,
            archivePrefix = {arXiv},
            eprinttype = {arxiv},
            eprint = {1509.01626},
            primaryClass = {cs},
            title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
            journal = {arXiv:1509.01626 [cs]},
            author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
            year = {2015},
        }

    License:
        Downstream variant of the Yelp Dataset Challenge dataset, usage of
        which is governed by the original user agreement, available
        (here)[https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset].

        "Yelp grants you a royalty-free, non-exclusive, revocable, 
        non-sublicensable, non-transferable, fully paid-up right and 
        license during the Term to use, access, and create derivative 
        works of the Data in electronic form for solely for non-commercial use."
    """
    DATA_DIR = 'data/YelpPolarity/'
    DATASET_NAME = 'YelpPolarity'
    DOWNLOAD_URL = 'https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz'
    FILE_NAMES = ['train.csv', 'test.csv', 'readme.txt']

    # check if we need to download
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not all([os.path.exists(DATA_DIR + f) for f in FILE_NAMES]):
        # delete any existing files
        for fn in FILE_NAMES:
            if os.path.exists(DATA_DIR + fn):
                os.remove(DATA_DIR + fn)

        # download the dataset
        download_dataset(DATASET_NAME, DATA_DIR, DOWNLOAD_URL)

        # reorganize files
        for fn in FILE_NAMES:
            os.rename(DATA_DIR + f'yelp_review_polarity_csv/{fn}', DATA_DIR + f'{fn}')
        os.rmdir(DATA_DIR + 'yelp_review_polarity_csv')

    # load both train and test data
    train_df = pd.read_csv(DATA_DIR + 'train.csv', header=None)
    test_df = pd.read_csv(DATA_DIR + 'test.csv', header=None)
    df = pd.concat([train_df, test_df])

    # rename columns
    df.rename(columns={0: 'label', 1: 'text'}, inplace=True)

    # check for rows with nan comment text
    idxs = df['text'].isna()
    if idxs.sum() > 0: df = df[~idxs]
    # collect features and labels
    y = df['label'].apply(lambda x: 1 if x == 2 else 0).values
    df = df.drop(columns=['label'])
    X = df['text'].values

    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    return X, y, (gps, gp_names)
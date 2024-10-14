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
        # category
        'food',
        'health',
        'music',
        'book',
        'movie',
        'tech',
        'cooking',
        'shirt',
        'fabric',
        'pants',
        'shoe',
        'exercise',
        # judgement
        'hype',
        'garbage',
        'terrible',
        'incredible',
        'love',
        'again',
        'star',
    ]
    groups_map = {
        keyword: np.where(features_df['text'].str.contains(keyword, regex=False))[0]
        for keyword in keywords
    }

    return groups_map


def load_AmazonPolarity(groups='default'):
    """
    Amazon Review Polarity dataset. Dataset documents reviews 
    of products sold on Amazon and associated user metadata.

    Input (x): Amazon review text.
    Label (y): 1 if review >= 4/5 stars, 0 if review <= 2/5 stars.

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

    Source of data:
        @inproceedings{mcauley2013hidden,
            author       = {Julian J. McAuley and
                        Jure Leskovec},
            title        = {Hidden factors and hidden topics: understanding rating dimensions
                            with review text},
            booktitle    = {RecSys},
            pages        = {165--172},
            publisher    = {{ACM}},
            year         = {2013}
        }

    License: 
        Data introduced by McAuley et al. and Stanford Network Analysis Project.
        We were not able to find a license for this dataset, though it was 
        generated with publicly available content from Internet Archive.
    """
    DATA_DIR = 'data/AmazonPolarity/'
    DATASET_NAME = 'AmazonPolarity'
    DOWNLOAD_URL = 'https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz'
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
            os.rename(DATA_DIR + f'amazon_review_polarity_csv/{fn}', DATA_DIR + f'{fn}')
        os.rmdir(DATA_DIR + 'amazon_review_polarity_csv')

    # load both train and test data
    train_df = pd.read_csv(DATA_DIR + 'train.csv', header=None)
    test_df = pd.read_csv(DATA_DIR + 'test.csv', header=None)
    df = pd.concat([train_df, test_df])

    # shuffle deterministically, take first 200k
    df = df.sample(frac=1, random_state=15)
    df = df.iloc[:400000]

    # rename columns
    df.rename(columns={0: 'label', 1: 'title', 2: 'text'}, inplace=True)

    # check for rows with nan comment text
    idxs = df['text'].isna() | df['title'].isna()
    if idxs.sum() > 0: df = df[~idxs]
    # collect features and labels
    y = df['label'].apply(lambda x: 1 if x == 2 else 0).values
    df.drop(columns=['label'], inplace=True)
    # absorb title into text
    df['text'] = df['title'] + ' ' + df['text']
    df.drop(columns=['title'], inplace=True)

    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    X = df['text'].values
    return X, y, (gps, gp_names)
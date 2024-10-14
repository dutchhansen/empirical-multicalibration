from dataloaders.utils.download_utils import download_dataset
import numpy as np
import pandas as pd
import torch
import os
import shutil

def groups_map(features_df, groups='default'):
    df = features_df
    groups_map = {
        'Male': np.where((df['male'] == 1))[0],
        'Female': np.where((df['female'] == 1))[0],
        'LGBTQ': np.where((df['LGBTQ'] == 1))[0],
        'Not_LGBTQ': np.where((df['LGBTQ'] == 0))[0],
        'Christian': np.where((df['christian'] == 1))[0],
        'Not_Christian': np.where((df['christian'] == 0))[0],
        'Muslim': np.where((df['muslim'] == 1))[0],
        'Not_Muslim': np.where((df['muslim'] == 0))[0],
        'Other_Religions': np.where((df['other_religions'] == 1))[0],
        'Black': np.where((df['black'] == 1))[0],
        'Not_Black': np.where((df['black'] == 0))[0],
        'White': np.where((df['white'] == 1))[0],
        'Not_White': np.where((df['white'] == 0))[0]
    }
    return groups_map


def load_CivilComments(groups='default'):
    """
    A shuffled variant of the CivilComments-WILDS dataset. The CivilComments dataset, 
    introduced by Borkan et al. (2019), contains 450,000 online comments annotated 
    for toxicity and identity mentions by crowdsourcing and majority vote. We use 
    the WILDS variant of this dataset, provided by Koh et al. (2021), though we shuffle 
    the predetermined training and test splits, and consider the task of prediction 
    whether a given comment is labeled toxic.

    Input (x): Comment.
    Label (y): 1 if comment is toxic, 0 otherwise.

    Website:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Original publication:
        @inproceedings{borkan2019nuanced,
            title={Nuanced metrics for measuring unintended bias with real data for text classification},
            author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
            booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
            pages={491--500},
            year={2019}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """
    DATA_DIR = 'data/CivilComments/'
    DATASET_NAME = 'CivilComments'
    DOWNLOAD_URL = 'https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/'
    FILE_NAMES = ['all_data_with_identities.csv']

    # check if we need to download
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not all([os.path.exists(DATA_DIR + f) for f in FILE_NAMES]):
        # delete any existing files/dirs
        files = os.listdir(DATA_DIR)
        for fn in files:
            if os.path.exists(DATA_DIR + fn):
                # if is file / dir
                if os.path.isfile(DATA_DIR + fn):
                    os.remove(DATA_DIR + fn)
                else:
                    shutil.rmtree(DATA_DIR + fn)

        # download the dataset
        download_dataset(DATASET_NAME, DATA_DIR, DOWNLOAD_URL)
        
        # delete unnecessary
        files = os.listdir(DATA_DIR)
        remove_files = [f for f in files if f not in FILE_NAMES]
        for fn in remove_files:
            if os.path.isfile(DATA_DIR + fn):
                os.remove(DATA_DIR + fn)
            else: 
                shutil.rmtree(DATA_DIR + fn)

    # load data
    df = pd.read_csv(DATA_DIR + 'all_data_with_identities.csv', index_col=0)
    # check for rows with nan comment text
    idxs = df['comment_text'].isna()
    if idxs.sum() > 0: df = df[~idxs]
    y = (df['toxicity'].values >= 0.5).astype(int)
    df = df.drop(columns=['toxicity'])

    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)
    
    # return only text for model
    X = df['comment_text'].values

    return X, y, (gps, gp_names)

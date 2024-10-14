from utils import href
from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files, req_urls
import pandas as pd
import numpy as np
import shutil
import os


def groups_map(features_df, groups='default'):
    df = features_df
    if groups == 'default':
        groups_map = {
            # age
            'Age 0-18': np.where(df['AGE'] <= 18)[0],
            'Age 19-34': np.where((df['AGE'] > 18) & (df['AGE'] <= 34))[0],
            'Age 35-50': np.where((df['AGE'] > 34) & (df['AGE'] <= 50))[0],
            'Age 51-64': np.where((df['AGE'] > 50) & (df['AGE'] <= 64))[0],
            'Age 65-79': np.where((df['AGE'] > 64) & (df['AGE'] <= 79))[0],
            # race
            'Not White': np.where(df['RACE=NW'])[0],
            # part of country
            'Northeast': np.where(df['REGION=1'])[0],
            'Midwest': np.where(df['REGION=2'])[0],
            'South': np.where(df['REGION=3'])[0],
            'West': np.where(df['REGION=4'])[0],
            # income
            'Poverty Category 1': np.where(df['POVCAT=1'])[0],
            'Poverty Category 2': np.where(df['POVCAT=2'])[0],
            'Poverty Category 3': np.where(df['POVCAT=3'])[0],
            'Poverty Category 4': np.where(df['POVCAT=4'])[0],
        }
    elif groups == 'alternate':
        groups_map = {
            # Age
            'Under 21': np.where(
                (df['AGE'] <= 21)
            )[0],
            'Middle Age': np.where(
                (df['AGE'] >= 40) & (df['AGE'] <= 60)
            )[0],
            'Senior Age': np.where(
                (df['AGE'] >= 65)
            )[0],

            # Sex
            'Sex = 1': np.where(
                (df['SEX=1'])
            )[0],
            'Sex = 2': np.where(
                (df['SEX=2'])
            )[0],

            # Race
            'White': np.where(
                (df['RACE=W'])
            )[0],

            # Military
            'Active Duty Group 1': np.where(
                (df['ACTDTY=1'])
            )[0],
            'Active Duty Group 2': np.where(
                (df['ACTDTY=2'])
            )[0],

            # Marriage
            'Marriage Group 1': np.where(
                (df['MARRY=1'])
            )[0],
            'Marriage Group 2': np.where(
                (df['MARRY=2'])
            )[0],

            # Pregnancy
            'Pregnancy Group 1': np.where(
                (df['PREGNT=1'])
            )[0],
            'Pregnancy Group 2': np.where(
                (df['PREGNT=2'])
            )[0],

            # Insurance
            'Insurance Group 1': np.where(
                (df['INSCOV=1'])
            )[0],
            'Insurance Group 2': np.where(
                (df['INSCOV=2'])
            )[0],
        }

    return groups_map


def load_MEPS_no_pov():
    return load_MEPS(drop_features=['POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4'])


def load_MEPS(drop_features=[], groups='default'):
    """
    Dataset comes from US Department of Health and Human Services. Documents
    healthcare utilization and expenditures of US citizens. 

    Input (x): Single survey response from family or individual.
    Label (y): 1 if healthcare utilization is greater than 10 visits, 0 otherwise.

    Website:
        official: https://meps.ahrq.gov/mepsweb/
        this variant: https://github.com/alangee/FaiR-N/tree/master

    Original source:
      Agency for Healthcare Research and Quality.
      original data files: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp

    This variant of dataset:
        @article{DBLP:journals/corr/abs-2010-06113,
            author       = {Shubham Sharma and
                            Alan H. Gee and
                            David Paydarfar and
                            Joydeep Ghosh},
            title        = {FaiR-N: Fair and Robust Neural Networks for Structured Data},
            journal      = {CoRR},
            volume       = {abs/2010.06113},
            year         = {2020}
        }
    
    License:
        Apache License, Version 2.0, January 2004
    """

    DATA_DIR = 'data/MEPS/'
    DATASET_NAME = 'MEPS'
    FILE_NAMES = req_files(DATASET_NAME)
    FILE_URLS = req_urls(DATASET_NAME)

    # check if we need to download
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # check if any file missing
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
        for url in FILE_URLS:
            download_dataset(DATASET_NAME, DATA_DIR, url)

    # read in all data
    df_train = pd.read_csv(DATA_DIR + 'meps_train_processed.csv').astype(float)
    df_test = pd.read_csv(DATA_DIR + 'meps_test_processed.csv').astype(float)
    df_lbl_train = pd.read_csv(DATA_DIR + 'mepsdata_train_labels.txt', delimiter=' ').astype(float)
    df_lbl_test = pd.read_csv(DATA_DIR + 'mepsdata_test_labels.txt', delimiter=' ').astype(float)
    # Remove added space from column names
    df_train.columns = df_train.columns.str.replace(' ', '')
    df_test.columns = df_test.columns.str.replace(' ', '')

    # join into one dataframe
    df = pd.concat([df_train, df_test], axis=0)
    df_lbl = pd.concat([df_lbl_train, df_lbl_test], axis=0)

    # define groups
    gm = groups_map(df, groups)

    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # record labels as 1 or 0
    y = df_lbl['UTILIZATION'].values

    # drop features
    df = df.drop(drop_features, axis=1)

    # encode categorical features using get_dummies
    X = df.values

    return X, y, (gps, gp_names)
from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files, req_urls
import shutil
import zipfile
import os
import pandas as pd
import numpy as np

def groups_map(features_df, groups='default'):
    df = features_df
    if groups == 'default':
        groups_map = {
            # jobs
            'Job = Management': np.where(df['job'] == 'management')[0],
            'Job = Technician': np.where(df['job'] == 'technician')[0],
            'Job = Entrepreneur': np.where(df['job'] == 'entrepreneur')[0],
            'Job = Blue-Collar': np.where(df['job'] == 'blue-collar')[0],
            'Job = Retired': np.where(df['job'] == 'retired')[0],
            # marital
            'Marital = Married': np.where(df['marital'] == 'married')[0],
            'Marital = Single': np.where(df['marital'] == 'single')[0],
            # education
            'Education = Primary': np.where(df['education'] == 'primary')[0],
            'Education = Secondary': np.where(df['education'] == 'secondary')[0],
            'Education = Tertiary': np.where(df['education'] == 'tertiary')[0],
            # housing
            'Housing = Yes': np.where(df['housing'] == 'yes')[0],
            'Housing = No': np.where(df['housing'] == 'no')[0],
            # age
            'Age < 30': np.where(df['age'] < 30)[0],
            '30 <= Age < 40': np.where((df['age'] >= 30) & (df['age'] < 40))[0],
            'Age >= 50': np.where((df['age'] >= 50))[0],
        }
    
    elif groups == 'alternate':
        groups_map = {
            # jobs and ages
            'Job = Management, Age < 50': np.where(
                (df['job'] == 'management') & 
                (df['age'] < 50)
            )[0],
            'Job = Technician, Age < 30': np.where(
                (df['job'] == 'technician') & 
                (df['age'] < 30)
            )[0],
            'Job = Entrepreneur, Age < 30': np.where(
                (df['job'] == 'entrepreneur') & 
                (df['age'] < 30)
            )[0],
            'Job = Blue-Collar, Age > 50': np.where(
                (df['job'] == 'blue-collar') & 
                (df['age'] > 50)
            )[0],
            'Job = Retired, Age > 80': np.where(
                (df['job'] == 'retired') & 
                (df['age'] > 80)
            )[0],
            
            # marital and education
            'Married, Education = Primary': np.where(
                (df['marital'] == 'married') & 
                (df['education'] == 'primary')
            )[0],
            'Single, Education = Tertiary': np.where(
                (df['marital'] == 'single') & 
                (df['education'] == 'tertiary')
            )[0],

            # housing and age
            'Housing = Yes, Age < 30': np.where(
                (df['housing'] == 'yes') & 
                (df['age'] < 30)
            )[0],
            'Housing = No, Age < 30': np.where(
                (df['housing'] == 'no') & 
                (df['age'] < 30)
            )[0],

            # age
            'Under 21': np.where(
                (df['age'] < 21)
            )[0],
            'Middle Age': np.where(
                (df['age'] >= 40) & 
                (df['age'] <= 60)
            )[0],
            'Senior Age': np.where(
                (df['age'] >= 65)
            )[0],
        }
    
    else:
        raise ValueError('Unknown group name')
    
    return groups_map


def load_BankMarketing_no_job():
    return load_BankMarketing(drop_features=['job'])


def load_BankMarketing(drop_features=[], groups='default'):
    '''
    Dataset documents phone calls made by Portuguese banking institution 
    during several marketing campaigns. The goal is to predict whether a a client
    will subscribe to a term deposit or not.

    Input (x): Single phone call.
    Label (y): A 'yes' or 'no' in 'y' column.

    Website:
        https://archive.ics.uci.edu/dataset/222/bank+marketing

    Original publication:
        @misc{misc_statlog_(german_credit_data)_144,
            author       = {Hofmann,Hans},
            title        = {{Statlog (German Credit Data)}},
            year         = {1994},
            howpublished = {UCI Machine Learning Repository},
            note         = {{DOI}: https://doi.org/10.24432/C5NC77}
        }

    License:
        This dataset is licensed under a Creative Commons Attribution 4.0 
        International (CC BY 4.0) license. This allows for the sharing and adaptation 
        of the datasets for any purpose, provided that the appropriate credit is given.
    '''

    DATA_DIR = 'data/BankMarketing/'
    DATASET_NAME = 'BankMarketing'
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

        # remove 'bank-additional.zip' file
        os.remove(DATA_DIR + 'bank-additional.zip')
        # unzip remaining file
        with zipfile.ZipFile(DATA_DIR + 'bank.zip', 'r') as z:
            z.extractall(DATA_DIR + 'bank_temp')
        # move 'bank-full.csv' to root
        shutil.move(DATA_DIR + 'bank_temp/bank-full.csv', DATA_DIR + 'bank-full.csv')
        # remove 'bank_temp' dir and 'bank.zip'
        shutil.rmtree(DATA_DIR + 'bank_temp')
        os.remove(DATA_DIR + 'bank.zip')

    df = pd.read_csv(DATA_DIR + 'bank-full.csv', sep=';')

    # handling missing values
    df.drop(columns=['day'], inplace=True)  # remove 'day' column
    df.replace(" ?", pd.NA, inplace=True)  # replace ' ?' with NA
    df.dropna(inplace=True)  # drop NA values
    df["age"] = df["age"].astype(int).apply(lambda x: round(x / 5) * 5)

    # define groups
    gm = groups_map(df, groups)

    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # record labels as 1 or 0
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).values
    df = df.drop(columns=['y'])

    # drop features
    df = df.drop(drop_features, axis=1)

    # encode categorical features using get_dummies
    categories = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    categorical = [c for c in categories if c in df.columns]
    X = pd.get_dummies(df, columns=categorical, drop_first=True)
    X = X.values.astype(float)

    return X, y, (gps, gp_names)
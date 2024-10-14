from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files, req_urls
import shutil
import os
import pandas as pd
import numpy as np

def groups_map(features_df, groups='default'):
    df = features_df
    if groups == 'default':
        groups_map = {
            'Male, Age < 30': np.where((df['SEX'] == 1) & (df['AGE'] < 30))[0],
            'Single': np.where((df['MARRIAGE'] == 2))[0],
            'Single, Age > 30': np.where((df['MARRIAGE'] == 2) & (df['AGE'] > 30))[0],
            'Female': np.where((df['SEX'] == 2))[0],
            'Female, Age > 65': np.where((df['SEX'] == 2) & (df['AGE'] > 65))[0],
            'Married, Age < 30': np.where((df['MARRIAGE'] == 1) & (df['AGE'] < 30))[0],
            'Married, Age > 60': np.where((df['MARRIAGE'] == 1) & (df['AGE'] > 60))[0],
            'Education = Other': np.where((df['EDUCATION'] == 4))[0],
            'Education = High School': np.where((df['EDUCATION'] == 3))[0],
            'Education = High School, Married': np.where((df['EDUCATION'] == 3) & (df['MARRIAGE'] == 1))[0],
            'Education = High School, Age > 40': np.where((df['EDUCATION'] == 3) & (df['AGE'] > 40))[0],
            'Education = University, Age < 25': np.where((df['EDUCATION'] == 2) & (df['AGE'] < 25))[0],
            'Female, Education = University': np.where((df['SEX'] == 2) & (df['EDUCATION'] == 2))[0],
            'Education = Graduate School': np.where((df['EDUCATION'] == 1))[0],
            'Female, Education = Graduate School': np.where((df['SEX'] == 2) & (df['EDUCATION'] == 1))[0],
        }
    elif groups == 'alternate':
        groups_map = {
            # relationship status and gender identity
            'Single, Male': np.where(
                (df['MARRIAGE'] == 2) & 
                (df['SEX'] == 1)
            )[0],
            'Single, Female': np.where(
                (df['MARRIAGE'] == 2) &
                (df['MARRIAGE'] == 2)
            )[0],

            # age groups
            'Under 21': np.where(
                (df['AGE'] < 21)
            )[0],
            'Young Adult': np.where(
                (df['AGE'] >= 21) & 
                (df['AGE'] < 30)
            )[0],
            'Middle Aged': np.where(
                (df['AGE'] >= 40) & 
                (df['AGE'] <= 60)
            )[0],
            'Senior Aged': np.where(
                (df['AGE'] >= 65)
            )[0],

            # education and gender identity
            'Education = High School, Female': np.where(
                (df['EDUCATION'] == 3) &
                (df['SEX'] == 2)
            )[0],
            'Education = University, Female': np.where(
                (df['EDUCATION'] == 2) &
                (df['SEX'] == 2)
            )[0],

            # education and relationship status
            'Education = High School, Single': np.where(
                (df['EDUCATION'] == 3) &
                (df['MARRIAGE'] == 2)
            )[0],
            'Education = High School, Married': np.where(
                (df['EDUCATION'] == 3) &
                (df['MARRIAGE'] == 1)
            )[0],
            'Education = University, Single': np.where(
                (df['EDUCATION'] == 2) &
                (df['MARRIAGE'] == 2)
            )[0],
            'Education = University, Married': np.where(
                (df['EDUCATION'] == 2) &
                (df['MARRIAGE'] == 1)
            )[0],
            'Education = Graduate, Single': np.where(
                (df['EDUCATION'] == 1) &
                (df['MARRIAGE'] == 2)
            )[0],
        }
    else:
        return ValueError('Invalid group name')

    return groups_map


def load_CreditDefault_no_edu():
    return load_CreditDefault(drop_features=['EDUCATION'])


def load_CreditDefault(drop_features=[], groups='default'):
    """
    UCI Default of Credit Card Clients dataset (termed “Credit Default” in our experiments). 
    Documents the partial credit histories of 30,000 Taiwanese individuals (Yeh, 2016). 
    We consider the task of predicting whether an individual will default on credit card debt, 
    given payment history and additional identity attributes.

    Input (x): Credit card payment history of individual.
    Label (y): 1 if individual defaults on credit card debt, 0 otherwise.

    Website:
        https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

    Original publication:
        @misc{misc_default_of_credit_card_clients_350,
            author       = {Yeh,I-Cheng},
            title        = {{Default of Credit Card Clients}},
            year         = {2016},
            howpublished = {UCI Machine Learning Repository},
            note         = {{DOI}: https://doi.org/10.24432/C55S3H}
        }

    License:
        This dataset is licensed under a Creative Commons Attribution 4.0 
        International (CC BY 4.0) license. This allows for the sharing and adaptation 
        of the datasets for any purpose, provided that the appropriate credit is given.
    
    """

    DATA_DIR = 'data/CreditDefault/'
    DATASET_NAME = 'CreditDefault'
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

    # Load data with pandas
    file_name = 'default of credit card clients.xls'
    df = pd.read_excel(DATA_DIR + file_name, header=1)

    # Extract Y and drop it from the dataframe
    y = df['default payment next month'].values
    df = df.drop(columns=['default payment next month'])

    # define groups
    gm = groups_map(df, groups)

    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # preprocess data
    # Drop ID column
    df = df.drop(columns=['ID'])
    # Bucket LIMIT_BAL column into 0, 1, 2 based on quantiles
    df['LIMIT_BAL'] = pd.qcut(df['LIMIT_BAL'], 3, labels=[0, 1, 2])
    # Bucket AGE column into 0, 1, 2 based on quantiles
    df['AGE'] = pd.qcut(df['AGE'], 3, labels=[0, 1, 2])
    # Bucket BILL_AMT1 through BILL_AMT6 columns into 0, 1, 2 based on quantiles
    for i in range(1, 7):
        df[f'BILL_AMT{i}'] = pd.qcut(df[f'BILL_AMT{i}'], 3, labels=[0, 1, 2])
    # Bucket PAY_AMT1 through PAY_AMT6 columns into 0, 1, 2 based on quantiles
    for i in range(1, 7):
        df[f'PAY_AMT{i}'] = pd.qcut(df[f'PAY_AMT{i}'], 3, labels=[0, 1, 2])

    # Drop features
    df = df.drop(columns=drop_features)

    # the following columns are categorical, so we can one-hot encode them
    categories = ['LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE', 
                                     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df = pd.get_dummies(df, columns=[c for c in categories if c not in drop_features])

    # Get features as numpy array
    X = df.values.astype(float)
    return X, y, (gps, gp_names)
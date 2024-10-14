from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files, req_urls
import wget
import os
import pandas as pd
import numpy as np
import shutil

def groups_map(features_df, groups='default', preprocessed=True):
    df = features_df
    if not preprocessed:
        gm = {
            # age
            'Age 25-34': np.where(df['applicant_age'] == '25-34')[0],
            'Male Age 35-44': np.where((df['applicant_sex'] == 1) & (df['applicant_age'] == '35-44'))[0],
            'Age < 25': np.where(df['applicant_age'] == '<25')[0],
            'Age > 74': np.where((df['applicant_age'] == '>74'))[0],
            'Age 35-44': np.where((df['applicant_age'] == '35-44'))[0],
            # race
            'Black': np.where((df['applicant_race-1'] == 3))[0],
            'Asian': np.where((df['applicant_race-1'] == 2))[0],
            # dwelling
            'Single Family': np.where((df['derived_dwelling_category'] == 'Single Family (1-4 Units):Site-Built'))[0],
            # gender
            'Male': np.where((df['applicant_sex'] == 1))[0],
            'Female': np.where((df['applicant_sex'] == 2))[0]}
    else:
        if groups == 'default':
            gm = {
                'Applicant Ethnicity: Hispanic or Latino': np.where((df['applicant_ethnicity'] == 1))[0],
                'Applicant Ethnicity: Not Hispanic or Latino': np.where((df['applicant_ethnicity'] == 2))[0],
                'Applicant Ethnicity: Not provided': np.where((df['applicant_ethnicity'] == 3))[0],
                'Applicant Sex: Female': np.where((df['applicant_sex'] == 2))[0],
                'Applicant Sex: Male': np.where((df['applicant_sex'] == 1))[0],
                'Co-Applicant Sex: Female': np.where((df['co_applicant_sex'] == 2))[0],
                'Co-Applicant Sex: Male': np.where((df['co_applicant_sex'] == 1))[0],
                'Applicant Race: Black': np.where((df['applicant_race_1'] == 3))[0],
                'Applicant Race: Asian': np.where((df['applicant_race_1'] == 2))[0],
                'Applicant Race: Native American or Alaskan': np.where((df['applicant_race_1'] == 1))[0],
                'Co-Applicant Race: Black': np.where((df['co_applicant_race_1'] == 3))[0],
                'Co-Applicant Race: Asian': np.where((df['co_applicant_race_1'] == 2))[0],
                'Co-Applicant Race: Native American or Alaskan': np.where((df['co_applicant_race_1'] == 1))[0]
            }
        
        elif groups == 'alternate':
            gm = {
                # Loan types
                'Loan Type 1': np.where(
                    (df['loan_type'] == 1)
                )[0],
                'Loan Type 2': np.where(
                    (df['loan_type'] == 2)
                )[0],
                'Loan Type 3': np.where(
                    (df['loan_type'] == 3)
                )[0],

                # HUD Median Family Income
                'HUD Median Family Income > 50k': np.where(
                    (df['hud_median_family_income'] > 50000)
                )[0],
                'HUD Median Family Income <= 50k': np.where(
                    (df['hud_median_family_income'] <= 50000)
                )[0],

                # Co Applicant
                'Has Co-Applicant': np.where(
                    (df['has_co_applicant'] == 1)
                )[0],

                # Agency
                'Agency = OCC': np.where(
                    (df['agency_code'] == 1)
                )[0],
                'Agency = FRS': np.where(
                    (df['agency_code'] == 2)
                )[0],
                'Agency = FDIC': np.where(
                    (df['agency_code'] == 3)
                )[0],
                'Agency = NCUA': np.where(
                    (df['agency_code'] == 5)
                )[0],
                'Agency = HUD': np.where(
                    (df['agency_code'] == 7)
                )[0],
                'Agency = CFPB': np.where(
                    (df['agency_code'] == 9)
                )[0],

                # Loan type
                'Loan Type = 1 to 4 Family': np.where(
                    (df['loan_type'] == 1)
                )[0],
                'Loan Type = Manufactured Housing': np.where(
                    (df['loan_type'] == 2)
                )[0],
                'Loan Type = Multi-Family': np.where(
                    (df['loan_type'] == 3)
                )[0],
            }

        else: 
            raise ValueError('Invalid groups argument.')
    
    return gm


def load_processed_HMDA(drop_features=[], groups='default'):
    """
    The HMDA (Home Mortgage Disclosure Act) dataset documents the US mortgage 
    applications, identity attributes of associated applicants, and the outcome 
    of these applications (Federal Financial Institutions Examination Council, 2017). 
    We use a 114,000-sample variant of this dataset given by Cooper et al. (2023), 
    and consider the task of predicting whether a 2017 application in the state of
    Texas was accepted.

    Input (x): Mortgage application.
    Label (y): 1 if the application received positive action, 0 otherwise.

    Website:
        this variant: https://github.com/pasta41/hmda?tab=readme-ov-file.

    Original dataset:
        @misc{ffiec2022housingdata,
            author = {{Federal Financial Institutions Examination Council}},
            institution={Consumer Financial Protection Bureau},
            title ={{HMDA Data Publication}},
            year = {2017},
            note = {Released due to the Home Mortgage Disclosure Act},
            url = {https://www.consumerfinance.gov/data-research/hmda/historic-data/}
        }

    This variant of dataset:
        @misc{cooper2023variance,
            title={{Variance, Self-Consistency, and Arbitrariness in Fair Classification}}, 
            author={A. Feder Cooper and Solon Barocas and Christopher De Sa and Siddhartha Sen},
            year={2023},
            eprint={2301.11562},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    
    License:
        MIT License. 2023.
    """

    DATA_DIR = 'data/preprocessed_HMDA_TX/'
    DATASET_NAME = 'HMDA'
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

        # extract files from subdirs in original zip
        # list existing paths 
        paths = []
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                paths.append(os.path.join(root, file))
        # extract files from subdirs
        for p in paths:
            shutil.move(p, DATA_DIR + p.split('/')[-1])
            
        # remove empty dir from download
        shutil.rmtree(DATA_DIR + 'data')

    
    X_path = 'data/preprocessed_HMDA_TX/2017-TX-features.csv'
    y_path = 'data/preprocessed_HMDA_TX/2017-TX-target.csv'
    demographics_path = 'data/preprocessed_HMDA_TX/2017-TX-protected.csv'
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    demographics_df = pd.read_csv(demographics_path)

    # only keep 150,000 samples (fixed seed for reproducibility)
    df = pd.concat([X_df, demographics_df, y_df], axis=1)
    df = df.sample(n=150000, random_state=42)

    # only save rows with action_taken in {1, 2, 3, 5, 7, 8}
    df = df[df['action_taken'].isin([1, 2, 3, 5, 7, 8])]
    # replace 1, 2, 8 in action_taken with 1
    df['action_taken'] = df['action_taken'].replace([1, 2, 8], 1)
    # replace 3, 4, 5, 6, 7 in action_taken with 0
    df['action_taken'] = df['action_taken'].replace([3, 5, 7], 0)

    # drop denial reason cols
    df = df.drop(columns=['denial_reason_1', 'denial_reason_2', 'denial_reason_3'])

    # collect gropus
    gm = groups_map(df, groups, preprocessed=True)
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # drop features for expressiveness experiment
    df = df.drop(drop_features, axis=1)

    # drop any column with <= 1 unique value, figure out the categorical columns
    cols = [col for col in df.columns if col not in ['action_taken']]
    categorical_cols = []
    for col in cols:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
        else:
            if df[col].nunique() < 10:
                categorical_cols.append(col)

    y = df['action_taken'].values
    df = df.drop(columns=['action_taken'])

    # one hot encode the categorical columns
    df = pd.get_dummies(df, columns=categorical_cols)
    X = df.values.astype(float)
    return X, y, (gps, gp_names)


def load_processed_HMDA_no_race():
    return load_processed_HMDA(drop_features=['population',
                                              'minority_population',
                                              'applicant_race_1', 
                                              'applicant_race_2',
                                              'applicant_race_3',
                                              'applicant_race_4',
                                              'co_applicant_race_1',
                                              'co_applicant_race_2',
                                              'co_applicant_race_3',
                                              'co_applicant_race_4'], 
                                              groups='default')


def load_HMDA(drop_features=[], groups='default'):
    """
    Data loading function for original dataset.
    """
    DATA_DIR = 'data/HMDA/'

    # check if HMDA directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # check if we need to download
    if not os.path.exists(DATA_DIR + 'HMDA_TX_NY.csv'):
        print('Downloading HMDA data from CFPB. File size is 442 MB, may take a couple minutes due to their serving speed.')
        url = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=TX,NY&years=2018&actions_taken=1,3'
        filename = wget.download(url)
        # save file to data/HMDA/HMDA.csv
        print('Downloaded HMDA data to', filename)
        os.rename(filename, DATA_DIR + 'HMDA_TX_NY.csv')
    
    # Load data with pandas
    df = pd.read_csv(DATA_DIR + 'HMDA_TX_NY.csv')
    
    pd.set_option('display.max_columns', None)

    # pring the number of rows with action taken = 1 and 3
    print('Number of rows with action taken = 1 (loan originated):', len(df[df['action_taken'] == 1]))
    print('Number of rows with action taken = 3 (denied):', len(df[df['action_taken'] == 3]))

    # Replace action taken = 3 with 0
    df['action_taken'] = df['action_taken'].replace(3, 0)    

    # Use only the following columns in X:
    used_variables = ['derived_race', 'loan_type', 'loan_purpose', 'conforming_loan_limit', 'derived_dwelling_category', 'applicant_sex', 'purchaser_type',
                      'applicant_race-1', 'reverse_mortgage', 'loan_amount', 'interest_rate', 'origination_charges', 'loan_term', 'property_value', 'applicant_age',
                      'county_code', 'tract_owner_occupied_units', 'preapproval', 'derived_msa-md', 'census_tract', 'tract_population',
                      'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 'tract_one_to_four_family_homes'
                      ]
    
    # has co-applicant
    # drop all rows with applicant age = 8888 or 9999
    df = df[(df['applicant_age'] != '8888') & (df['applicant_age'] != '9999')]

    # record labels
    y = df['action_taken'].values
    df = df.drop(columns=['action_taken'])

    # fill in NaNs with the average value for each column
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # drop all columns not in used_variables
    df = df[used_variables]

    # define groups
    gm = groups_map(df, groups, preprocessed=False)
    gps, group_names = [], []
    for group in gm:
        gps.append(gm[group])
        group_names.append(group)

    # Replace all 'Exempt' in the loan_term column with -1
    # df['loan_term'] = df['loan_term'].replace('Exempt', -1)
    # Drop any row with an entry of 'Exempt' in any column
    df = df[~df.isin(['Exempt']).any(axis=1)]

    # drop features
    df = df.drop(drop_features, axis=1)

    non_categorical_cols = ['loan_amount', 'interest_rate', 'origination_charges', 'loan_term', 'property_value',
                            'tract_owner_occupied_units', 'derived_msa-md', 'tract_population',
                            'tract_minority_population_percent', 'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage', 
                            'tract_one_to_four_family_homes', 'census_tract', 'county_code']
    categories = [col for col in df.columns if col not in non_categorical_cols]
    
    # one hot encode the categorical columns
    print('One hot encoding categorical columns')
    df = pd.get_dummies(df, columns=[c for c in categories if c not in drop_features])
    X = df.values.astype(float)
    
    return X, y, (gps, group_names)

from folktables import ACSDataSource, BasicProblem, adult_filter
import numpy as np
import os

def groups_map(features_df, groups='default'):
    '''
    Available features for defining groups:
        'AGEP':     a
        'COW':      a
        'SCHL':     a
        'MAR':      a
        'OCCP':     a
        'POBP':     a
        'RELP':     a
        'WKHP':     a
        'SEX':      a
        'RAC1P':    a
    '''
    default_groups = ["Black_Adults", "Black_Females", "Women", "Never_Married",
                      "American_Indian", "Seniors", "White_Women", "Multiracial",
                      "White_Children", "Asian"]               
    race_groups = {
        "race-{0}".format(i): np.where(features_df['RAC1P'] == i)[0] 
        for i in range(1,10)
        }
    groups_map = {
        "Black_Adults": np.where(
            (features_df['RAC1P'] == 2) & 
            (features_df['AGEP'] >= 18) & 
            (features_df['AGEP'] <= 99)
            )[0],
        "Black_Females": np.where(
            (features_df['SEX'] == 2) & 
            (features_df['RAC1P'] == 2)
            )[0],
        "Women": np.where(
            (features_df['SEX'] == 2)
            )[0],
        "Never_Married": np.where(
            (features_df['MAR'] == 5)
            )[0],
        "American_Indian": np.where(
            (features_df['RAC1P'] == 3)
            )[0],
        "Seniors": np.where(
            (features_df['AGEP'] >= 65)
            )[0],
        "White_Women": np.where(
            (features_df['SEX'] == 2) & 
            (features_df['AGEP'] >= 18) & 
            (features_df['RAC1P'] == 1)
            )[0],
        "Multiracial": np.where(
            (features_df['RAC1P'] == 9)
            )[0],
        "White_Children": np.where(
            (features_df['AGEP'] < 18) & 
            (features_df['RAC1P'] == 1)
            )[0],
        "Asian": np.where(
            (features_df['RAC1P'] == 6)
            )[0],
        **race_groups
        }
    alternate = {
        "Associates_Degree_Male": np.where(
            (features_df['SCHL'] == 20) & 
            (features_df['SEX'] == 1)
        )[0],
        "Associates_Degree_Female": np.where(
            (features_df['SCHL'] == 20) & 
            (features_df['SEX'] == 2)
        )[0],
        "Divorced_Female": np.where(
            (features_df['MAR'] == 3) & 
            (features_df['SEX'] == 2)
        )[0],
        "Under_Part_Time": np.where(
            (features_df['WKHP'] < 20)
        )[0],
        "Part_Time": np.where(
            (features_df['WKHP'] <= 35)
        )[0],
        "Full_Time": np.where(
            (features_df['WKHP'] >= 40)
        )[0],
        "Over_Full_Time": np.where(
            (features_df['WKHP'] >= 60)
        )[0],
        "Not_White": np.where(
            (features_df['RAC1P'] != 1)
        )[0],
        "Government_Employee": np.where(
            (features_df['COW'] == 3) |
            (features_df['COW'] == 4) |
            (features_df['COW'] == 5)
        )[0],
        "Private_Employee": np.where(
            (features_df['COW'] != 3) &
            (features_df['COW'] != 4) &
            (features_df['COW'] != 5)
        )[0],
        "Under_21": np.where(
            (features_df['AGEP'] < 21)
        )[0],
        "Middle_Aged": np.where(
            (features_df['AGEP'] >= 40) &
            (features_df['AGEP'] <= 60)
        )[0],
    }

    if groups == 'default':
        return {gp: groups_map[gp] for gp in default_groups}
    elif groups == 'all':
        return groups_map
    elif groups == 'alternate':
        return alternate
    else:
        raise ValueError("Invalid group type")


def load_ACSIncome_no_race(states=['CA']):
    '''
    ACSIncome dataset, without race features.
    Race-dependent groups added, however.

    We do not use this function to produce results in the paper.
    '''
    return load_ACSIncome(states, drop_features=['RAC1P'],
                          groups='all')


def load_ACSIncome(states=['CA'], drop_features=[], groups='default'):
    '''
    While this dataset is not considered in the paper, 
    we provide it here for comparison with prior works.

    Dataset provides income data and demographic information 
    about US citizens. This data comes from the American Community 
    Survey (ACS) Public Use Microdata Sample (PUMS) files, which are managed 
    by the US Census Bureau. Our paper studies the 2018 data for California, 
    though more data is available.
    
    Input (x): Single individual.
    Label (y): Whether individual makes more than $50,000 a year.

    Website:
        https://github.com/socialfoundations/folktables

    Original publication:
        @article{ding2021retiring,
            title={Retiring Adult: New Datasets for Fair Machine Learning},
            author={Ding, Frances and Hardt, Moritz and Miller, John and Schmidt, Ludwig},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            year={2021}
        }

    License:
        While Folktables provides API for downloading ACS data, usage of this data 
        is governed by the terms of use provided by the Census Bureau. 
        For more information, see https://www.census.gov/data/developers/about/terms-of-service.html.
    '''
    income_feature_list = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 
                           'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P' ]

    state_str = ""
    for state in states:
        state_str += state + '_'
    
    DIR = "data/ACS/{0}/".format(state_str[:-1])
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir=DIR)

    dl = True
    # check if we need to download
    if os.path.exists(DIR):
        dl = False

    state_data = data_source.get_data(states=states, download=dl)

    # Income parsed by threshold 50000, for convenience
    ACSIncome_binary = BasicProblem(
        features=income_feature_list,
        target='PINCP',
        target_transform=lambda x: x > 50000,    
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1))

    # filter groups from panda dataframe
    features_df, targets_df, _ = ACSIncome_binary.df_to_pandas(state_data)
    gm = groups_map(features_df, groups)

    # record in list
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # drop features
    features_df = features_df.drop(drop_features, axis=1)

    # return data
    X = features_df.values
    y = targets_df.values.reshape(-1)

    return X, y, (gps, gp_names)
import pandas as pd
import numpy as np


def groups_map(features_df, groups='default'):   
    groups_features = [
            # "age",
            # "workclass",
            # "education",
            # "marital-status",
            "occupation",
            # "relationship",
            "race",
            "sex",
        ]
         
    groups_map = {
        f'{f}-{v}': np.where(features_df[f] == v)[0] 
        for f in groups_features 
        for v in features_df[f].unique()
        }

    return groups_map


def load_AdultIncome(states=['CA'], drop_features=[], groups='default'):
    '''
    While this dataset is not considered in the paper, 
    we provide it here for comparison with prior works.

    Dataset provides income data and demographic information 
    about US citizens from the 1994 Census database.

    Input (x): Single individual.
    Label (y): Whether individual makes more than $50,000 a year.

    Website:
        https://archive.ics.uci.edu/dataset/2/adult

    Original publication:
        @misc{misc_adult_2,
            author       = {Becker,Barry and Kohavi,Ronny},
            title        = {{Adult}},
            year         = {1996},
            howpublished = {UCI Machine Learning Repository},
            note         = {{DOI}: https://doi.org/10.24432/C5XW20}
        }

    License:
        This dataset is licensed under a Creative Commons Attribution 4.0 
        International (CC BY 4.0) license. This allows for the sharing and adaptation 
        of the datasets for any purpose, provided that the appropriate credit is given.
    '''
    
    # column names for the dataset
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    target = "income"  # Last attribute

    # loading the dataset from the UCI repository
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    df = pd.read_csv(url, header=None, names=column_names)

    # handling missing values
    df.replace(" ?", pd.NA, inplace=True)  # replace ' ?' with NA
    df.dropna(inplace=True)  # drop NA values

    # separate the predictors (features) and the target (label)
    features = df.drop(target, axis=1)
    target = df[target].map({" <=50K": 0, " >50K": 1})

    # groups
    gm = groups_map(features, groups)
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # convert categorical features
    features = pd.get_dummies(features)
    X = features.values
    y = target.values

    return X, y, (gps, gp_names)
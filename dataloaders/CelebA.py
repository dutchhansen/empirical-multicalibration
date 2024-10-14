from utils import href
from dataloaders.utils.download_utils import download_dataset
import numpy as np
import pandas as pd
import torch
import os
import shutil

def groups_map(features_df, groups='default'):
    df = features_df
    groups_map = {
        "Male": np.where((df['Male'] == 1))[0],
        "Female": np.where((df['Male'] == 0))[0],
        "Arched_Eyebrows": np.where((df['Arched_Eyebrows'] == 1))[0],
        "Bangs": np.where((df['Bangs'] == 1))[0],
        "Big_Lips": np.where((df['Big_Lips'] == 1))[0],
        "Chubby": np.where((df['Chubby'] == 1))[0],
        "Double_Chin": np.where((df['Double_Chin'] == 1))[0],
        "Eyeglasses": np.where((df['Eyeglasses'] == 1))[0],
        "High_Cheekbones": np.where((df['High_Cheekbones'] == 1))[0],
        "Mouth_Slightly_Open": np.where((df['Mouth_Slightly_Open'] == 1))[0],
        "Oval_Face": np.where((df['Oval_Face'] == 1))[0],
        "Pale_Skin": np.where((df['Pale_Skin'] == 1))[0],
        "Receding_Hairline": np.where((df['Receding_Hairline'] == 1))[0],
        "Smiling": np.where((df['Smiling'] == 1))[0],
        "Straight_Hair": np.where((df['Straight_Hair'] == 1))[0],
        "Wavy_Hair": np.where((df['Wavy_Hair'] == 1))[0],
        "Wearing_Hat": np.where((df['Wearing_Hat'] == 1))[0],
        "Young": np.where((df['Young'] == 1))[0]}
    return groups_map


def load_CelebA(groups='default'):
    """
    A variant of the CelebA dataset.
    To download, visit: https://worksheets.codalab.org/rest/bundles/0xfe55077f5cd541f985ebf9ec50473293/contents/blob/

    Input (x): Images of celebrity faces, cropped and centered.
    Label (y): Binary. 1 if the celebrity in the image has blond hair, 0 otherwise.
    Metadata: Each image annotated with several attributes.

    Website:
        http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Original publication:
        @inproceedings{liu2015faceattributes,
            title = {Deep Learning Face Attributes in the Wild},
            author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
            booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
            month = {December},
            year = {2015}
        }

    This variant of the dataset is identical to that of the WILDS benchmark,
    though we split the data randomly into train and test sets.
        @inproceedings{sagawa2019distributionally,
            title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
            author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
            booktitle = {International Conference on Learning Representations},
            year = {2019}
        }

    License:
        The creators of this dataset do not provide a license, though they encourage its 
        use for non-commercial research purposes only. The WILDS version of the dataset was 
        originally downloaded from Kaggle. https://www.kaggle.com/jessicali9530/celeba-dataset
    """
    DATA_DIR = 'data/CelebA/'
    DATASET_NAME = 'CelebA'
    DOWNLOAD_URL = 'https://worksheets.codalab.org/rest/bundles/0xfe55077f5cd541f985ebf9ec50473293/contents/blob/'
    FILE_NAMES = ['list_attr_celeba.csv', 'img_align_celeba/']

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
    df = pd.read_csv(DATA_DIR + 'list_attr_celeba.csv')
    df.replace(to_replace=-1, value=0, inplace=True)

    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)
    
    # save features and labels
    X = df['image_id'].values
    y = df['Blond_Hair'].values

    return X, y, (gps, gp_names)
from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files
import shutil
import numpy as np
import pandas as pd
import os

def groups_map(features_df, groups='default'):
    df = features_df
    location_col = df['place_filename'].str.split('/').str[2]
    groups_map = {
        "Water bg": np.where((df['place'] == 1))[0],
        "Land bg": np.where((df['place'] == 0))[0],
        "ocean": np.where((location_col == 'ocean'))[0],
        "non_ocean": np.where((location_col != 'ocean'))[0],
        "lake": np.where((location_col == 'lake'))[0],
        "non_lake": np.where((location_col != 'lake'))[0],
        "bamboo_forest": np.where((location_col == 'bamboo_forest'))[0],
        "non_bamboo_forest": np.where((location_col != 'bamboo_forest'))[0],
        "forest": np.where((location_col == 'forest'))[0],
        "non_forest": np.where((location_col != 'forest'))[0],
    }
    return groups_map


def load_WaterBirds(groups='default'):
    """
    The Waterbirds dataset.
    We do not experiment with this dataset in our paper, 
    though we provide it here for convenience.

    Input (x): Images of birds against various backgrounds that have already been cropped and centered.
    Label (y): Binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.
    Metadata: Each image is annotated with whether the background is a land or water background.

    Original publication:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
        @techreport{WahCUB_200_2011,
        	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        	Year = {2011}
        	Institution = {California Institute of Technology},
        	Number = {CNS-TR-2011-001}
        }
        @article{zhou2017places,
            title = {Places: A 10 million Image Database for Scene Recognition},
            author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
            journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
            year = {2017},
            publisher = {IEEE}
        }

    License:
        Usage restricted to non-commercial research and educational purposes.
    """
    DATA_DIR = 'data/WaterBirds/'
    DATASET_NAME = 'WaterBirds'
    DOWNLOAD_URL = 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/'
    FILE_NAMES = req_files(DATASET_NAME)

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
        download_dataset(DATASET_NAME, DATA_DIR, DOWNLOAD_URL)
    
    # load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'),
                     index_col=0, dtype={'patient': 'str'})

    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)
    
    # save features and labels
    X = df['img_filename'].values
    y = df['y'].values

    return X, y, (gps, gp_names)
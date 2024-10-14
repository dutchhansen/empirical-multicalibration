from dataloaders.utils.download_utils import download_dataset
from configs.downloads import req_files
import shutil
import numpy as np
import pandas as pd
import os

def groups_map(features_df, groups='default'):
    df = features_df

    # 5 possible hospitals
    hospitals = {
        f'Hospital = {i}': np.where((df['center'] == i))[0]
        for i in range(5)
    }

    # 50 possible slides, so sample every 4
    slide_idxs = list(range(0, 50, 4))
    slides = {
        f'Slide = {i}': np.where((df['slide'] == i))[0]
        for i in slide_idxs
    }

    groups_map = {**hospitals, **slides}
    return groups_map


def load_Camelyon17(groups='default'):
    """
    The CAMELYON17-WILDS histopathology dataset.
    This is a modified version of the original CAMELYON17 dataset.
    To download, visit: https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/

    Input (x): 96x96 image patches extracted from histopathology slides.
    Label (y): Binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.
    Metadata: Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
                and the slide it came from (integer from 0 to 49).

    Website: https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
            title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
            author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
            journal={IEEE transactions on medical imaging},
            volume={38},
            number={2},
            pages={550--560},
            year={2018},
            publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """
    DATA_DIR = 'data/Camelyon17/'
    DATASET_NAME = 'Camelyon17'
    DOWNLOAD_URL = 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/'
    FILE_NAMES = req_files(DATASET_NAME)

    # Download
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
        # this download link gives no unnecessary files
        download_dataset(DATASET_NAME, DATA_DIR, DOWNLOAD_URL)

    # load data
    df = pd.read_csv(
            os.path.join(DATA_DIR, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

    # generate image paths
    df['image_path'] = [
        f'patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
        for patient, node, x, y in 
        df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
    
    # define groups
    gm = groups_map(df, groups)
    # record groups and names
    gps, gp_names = [], []
    for group in gm:
        gps.append(gm[group])
        gp_names.append(group)

    # save features and labels
    X = df['image_path'].values
    y = df['tumor'].values

    return X, y, (gps, gp_names)


from dataloaders.Adult import load_AdultIncome
from dataloaders.ACS import load_ACSIncome, load_ACSIncome_no_race
from dataloaders.CreditDefault import load_CreditDefault, load_CreditDefault_no_edu
from dataloaders.HMDA import load_processed_HMDA, load_processed_HMDA_no_race
from dataloaders.BankMarketing import load_BankMarketing, load_BankMarketing_no_job
from dataloaders.MEPS import load_MEPS, load_MEPS_no_pov
from dataloaders.CivilComments import load_CivilComments
from dataloaders.YelpPolarity import load_YelpPolarity
from dataloaders.AmazonPolarity import load_AmazonPolarity
from dataloaders.CelebA import load_CelebA
from dataloaders.WaterBirds import load_WaterBirds
from dataloaders.Camelyon17 import load_Camelyon17
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE as imbl_smote
from prettytable import PrettyTable
import pandas as pd
import numpy as np
import torchtext


class Dataset:
    def __init__(self, dataset_name, 
                 split={'train': 0.6, 'val': 0.2, 'test': 0.2},
                 groups='default',
                 scale=False,
                 class_balance=False,
                 val_split_seed=43, 
                 min_group_size=0.005,
                 verbose=True):
        '''
        Parameters:
            :dataset_name: str, name of the dataset to load
            :split: dict, split fractions for train, val, test
            :groups: str, name of the group collection to use
            :scale: bool, whether to scale the data to mean 0, std 1
            :class_balance: bool, whether to balance the classes in the training data using SMOTE
            :val_split_seed: int, seed for splitting the training data into train and validation
            :min_group_size: float, minimum fraction of entire dataset that a group must have to 
                            be included in the subgroup metrics
            :verbose: bool, whether to print group information
        '''
        self.name = dataset_name
        self.min_group_size = min_group_size
        self.val_split_seed = val_split_seed
        self.train_test_val_split = split
        self.is_text = False

        # Determines if we scale data (normalize to mean 0, std 1) on a by dataset basis
        self.scale = scale
        self.class_balance = class_balance
        load_data_fn = None
        
        # standard datasets
        if self.name == 'ACSIncome':
            load_data_fn = load_ACSIncome
        elif self.name == 'CreditDefault':
            load_data_fn = load_CreditDefault
        elif self.name == 'HMDA':
            load_data_fn = load_processed_HMDA
        elif self.name == 'BankMarketing':
            load_data_fn = load_BankMarketing
        elif self.name == 'MEPS':
            load_data_fn = load_MEPS
        elif self.name == 'AdultIncome':
            load_data_fn = load_AdultIncome
        # expressiveness experiments
        elif self.name == 'ACSIncome_no_race':
            load_data_fn = load_ACSIncome_no_race
        elif self.name == 'CreditDefault_no_edu':
            load_data_fn = load_CreditDefault_no_edu
        elif self.name == 'BankMarketing_no_job':
            load_data_fn = load_BankMarketing_no_job
        elif self.name == 'MEPS_no_pov':
            load_data_fn = load_MEPS_no_pov
        elif self.name == 'HMDA_no_race':
            load_data_fn = load_processed_HMDA_no_race
        # language datasets
        elif self.name == 'CivilComments':
            self.is_text = True
            load_data_fn = load_CivilComments
        elif self.name == 'YelpPolarity':
            self.is_text = True
            load_data_fn = load_YelpPolarity
        elif self.name == 'AmazonPolarity':
            self.is_text = True
            load_data_fn = load_AmazonPolarity
        # image datasets
        elif self.name == 'CelebA':
            load_data_fn = load_CelebA
        elif self.name == 'WaterBirds':
            load_data_fn = load_WaterBirds
        elif self.name == 'Camelyon17':
            load_data_fn = load_Camelyon17
        else:
            raise ValueError('Unknown dataset name')
        
        self.X, self.y, (self.groups, self.group_names) = load_data_fn(groups=groups)
        
        # scale or class balance data
        self._preprocess_data()
        self._check_data_leakage()
        
        # print groups and their sizes
        if verbose: print(groups_info_str(self.X, self.y, self.groups, self.group_names))


    def vocab(self, tokenizer, max_len, min_freq):
        """
        Returns vocabular for text datasets.
        """
        assert self.is_text == True, 'Vocab method only associated with language datasets.'
        tokenized = pd.Series(self.X).apply(lambda x: tokenizer(x)[:max_len])
        v = torchtext.vocab.build_vocab_from_iterator(tokenized, 
                                                      min_freq=min_freq, 
                                                      specials=['<pad>', '<unk>'])
        v.set_default_index(v['<unk>'])
        return v
        

    def train_calibration_split(self, frac, train_overlap=0, seed=43):
        """
        Parameters
            :frac: float, fraction of the training data to use for calibration
            :train_overlap: float, fraction of the training data to include in the calibration set
            :seed: int, seed for splitting the training data into calibration and training data
        """
        (
            self.calibration_idxs, 
            self.groups_calib, 
            self.train_idxs_no_calib, 
            self.groups_train_no_calib
        ) = split_data(self.train_idxs, self.groups, 
                       int(len(self.train_idxs) * frac), seed=seed)
        
        # update calibration and training data
        self.X_calib = self.X[self.calibration_idxs]
        self.y_calib = self.y[self.calibration_idxs]
        self.X_train_no_calib = self.X[self.train_idxs_no_calib]
        self.y_train_no_calib = self.y[self.train_idxs_no_calib]

        # if desired, allow overlap between the training and calibration data
        if train_overlap > 0:
            print(f"Introducing overlap between train / calibration data: {train_overlap:.2f}")

            # draw some {train_overlap}-fraction of training data
            np.random.seed(seed)
            num_samples = int(len(self.train_idxs_no_calib) * train_overlap)
            tc_idxs = np.random.choice(self.train_idxs_no_calib, num_samples, replace=False)

            # update calib set
            # note that reindex_group assumes sorted indices, so sort here
            self.calibration_idxs = sorted(np.concatenate([self.calibration_idxs, tc_idxs]))
            self.X_calib = self.X[self.calibration_idxs]
            self.y_calib = self.y[self.calibration_idxs]

            # update groups
            self.groups_calib = []
            for group in self.groups:
                g_idxs = reindex_group(self.calibration_idxs, group)
                self.groups_calib.append(g_idxs)

        return self.X_train_no_calib, self.y_train_no_calib, self.groups_train_no_calib, self.X_calib, self.y_calib, self.groups_calib


    def _preprocess_data(self):
        """
        Handle assumptions on the data.
            1. Remove groups with lesss than min_group_size of entire data
            2. Require binary classification
            3. Set fixed test seed
            4. Ensure valid split fractions
            5. Split data into train, val, test
            6. Scale or balance data
        """

        # delete groups with less than min_group_size of entire data
        groups_temp, group_names_temp = [], []
        for group, name in zip(self.groups, self.group_names):
            if len(group) > self.min_group_size * len(self.X):
                groups_temp.append(group)
                group_names_temp.append(name)
        self.groups = groups_temp
        self.group_names = group_names_temp
        self.group_info = {i: self.group_names[i] for i in range(len(self.group_names))}
        self.num_features = self.X.shape[1] if len(self.X.shape) > 1 else None

        # (We only support 2 classes for now)
        self.num_classes = len(np.unique(self.y))
        if self.num_classes != 2:
            raise ValueError('Only binary classification supported')
        
        # construct train, val, test splits
        # always deterministically construct the test split.
        self.test_split_seed = 42

        # check for valid split
        if self.train_test_val_split['test'] + self.train_test_val_split['val'] + self.train_test_val_split['train'] > 1:
            raise ValueError('Split fractions must be at most 1')
        
        self.test_size = self.train_test_val_split['test']
        self.val_size = self.train_test_val_split['val']
        self.train_size = self.train_test_val_split['train']

        self.n = len(self.X)
        (
            self.test_idxs, 
            self.groups_test, 
            train_val_idxs, 
            _
        ) = split_data(np.arange(self.n), self.groups, 
                       int(self.n * self.test_size), self.test_split_seed)
        self.X_test = self.X[self.test_idxs]
        self.y_test = self.y[self.test_idxs]

        (
            self.val_idxs, 
            self.groups_val, 
            self.train_idxs, 
            self.groups_train
        ) = split_data(train_val_idxs, self.groups, 
                       int(self.n * self.val_size), self.val_split_seed)
        self.X_val = self.X[self.val_idxs]
        self.y_val = self.y[self.val_idxs]

        self.X_train = self.X[self.train_idxs]
        self.y_train = self.y[self.train_idxs]

        if self.scale:
            self.X_train, self.X_val, self.X_test = self._scale_data()
            print("Scaling dataset.")
        if self.class_balance:
            print("Balancing classes in training data.")
            self.X_train, self.y_train, self.groups_train = self._smote()

    def _scale_data(self):
        """Scale the data."""

        # scale data using sklearn StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _smote(self):
        """Balance the classes using SMOTE."""

        # count number of 0s and 1s
        lbl_cnt_0 = np.sum(self.y_train == 0)
        lbl_cnt_1 = np.sum(self.y_train == 1)
        lbl_tot = len(self.y_train)

        # invoke SMOTE to balance the classes
        sm = imbl_smote()
        balanced_idxs, _ = sm.fit_resample(np.arange(len(self.y_train)).reshape(-1, 1), self.y_train)
        X_train_balanced = self.X_train[balanced_idxs.flatten()]
        y_train_balanced = self.y_train[balanced_idxs.flatten()]
        groups_train_balanced = [reindex_group(balanced_idxs.flatten(), group) for group in self.groups_train]

        # record changes
        space = ' ' * 8
        smt_lbl_cnt_0 = np.sum(self.y_train == 0)
        smt_lbl_cnt_1 = np.sum(self.y_train == 1)
        smt_lbl_tot = len(self.y_train)
        
        # print
        print('*' * 20)
        print(f"SMOTE called to balance classes in training data.")
        # Calculate proportions for original label distribution
        orig_lbl_prop_0 = lbl_cnt_0 / lbl_tot
        orig_lbl_prop_1 = lbl_cnt_1 / lbl_tot
        # Prepare strings for original and SMOTE-adjusted proportions to ensure alignment
        orig_prop_0_str = f"{orig_lbl_prop_0:.2%}"
        orig_prop_1_str = f"{orig_lbl_prop_1:.2%}"
        smote_prop_0_str = f"{smt_lbl_cnt_0/smt_lbl_tot:.2%}"
        smote_prop_1_str = f"{smt_lbl_cnt_1/smt_lbl_tot:.2%}"
        # Calculate padding for alignment
        max_len = max(len(orig_prop_0_str), len(orig_prop_1_str), len(smote_prop_0_str), len(smote_prop_1_str))
        # Print with aligned arrows
        print(f"{space}{'Class 0:':<20} (orig) {orig_prop_0_str:>{max_len}} -> (SMOTE) {smote_prop_0_str:>{max_len}}")
        print(f"{space}{'Class 1:':<20} (orig) {orig_prop_1_str:>{max_len}} -> (SMOTE) {smote_prop_1_str:>{max_len}}")
        print(f"{space}{'# samples:':<20} (orig) {lbl_tot:>{max_len}} -> (SMOTE) {smt_lbl_tot}")
        print('*' * 20)

        return X_train_balanced, y_train_balanced, groups_train_balanced

    def _check_data_leakage(self):
        """Check for overlap in splitting"""
        # check for overlap between train and test indices
        overlap = np.intersect1d(self.train_idxs, self.test_idxs)
        if len(overlap) > 0:
            raise ValueError('Data leakage: Overlap between train and test')
        # train and val
        overlap = np.intersect1d(self.train_idxs, self.val_idxs)
        if len(overlap) > 0:
            raise ValueError('Data leakage: Overlap between train and val')
        # val and test
        overlap = np.intersect1d(self.val_idxs, self.test_idxs)
        if len(overlap) > 0:
            raise ValueError('Data leakage: Overlap between val and test')
        
    def groups_info_df(self):
        return groups_info_df(self.X, self.y, self.groups, self.group_names)
    


def split_data(idxs, groups, num_in_p1, seed):
    """
    Randomly split data into two parts, with split_frac as the fraction of the first part.
    Also split the groups accordingly, and reindex the groups to match the new split data.
    """
    np.random.seed(seed)
    p1_idxs = np.random.choice(idxs, num_in_p1, replace=False)
    p2_idxs = np.setdiff1d(idxs, p1_idxs)
    p1_idxs = np.sort(p1_idxs)
    p2_idxs = np.sort(p2_idxs)

    p1_groups = []
    p2_groups = []
    # Now split the groups
    for group in groups:
        p1_groups.append(reindex_group(p1_idxs, group))
        p2_groups.append(reindex_group(p2_idxs, group))
    
    return p1_idxs, p1_groups, p2_idxs, p2_groups


def reindex_group(idxs, group_idxs):
    """
    Reindex the group indices to match the new split data.
    Take 0-indexed location of group_idxs in idxs.
        For example, if idxs = [1, 2, 3, 4] and 
        group_idxs = [2, 4], we return [1, 3].
    Note that this function assumes the passed in idxs are
    already sorted, as otherwise the returned group indices will
    not be correct for the data chosen with unsorted indices.
    """
    # verify assumptions
    assert (idxs == np.sort(idxs)).all(), 'idxs must be sorted'

    new_group_idxs = []
    sorted_idxs = idxs
    sorted_group_idxs = np.sort(group_idxs)
    i = j = 0
    while j < len(sorted_idxs) and i < len(sorted_group_idxs):
        if sorted_group_idxs[i] == sorted_idxs[j]:
            new_group_idxs.append(j)
            i += 1
            j += 1
        elif sorted_group_idxs[i] < sorted_idxs[j]:
            i += 1
        else:
            j += 1
    
    return new_group_idxs


def groups_info_df(X, y, groups, group_names):
    '''
    Get group information as a pandas dataframe.
    '''
    data = []

    N_samples = len(X)

    for i, group in enumerate(groups):
        size = len(group)
        frac = size / len(X)
        if size == 0: y_mean = 0.5
        else: y_mean = np.mean(y[group])
        data.append([group_names[i], size, f'{frac:.4f}', f'{y_mean:.4f}'])
    
    data.append(['Dataset', N_samples, f'{N_samples / len(X):.4f}', f'{np.mean(y):.4f}'])
    df = pd.DataFrame(data, columns=['group name', 'n samples', 'fraction', 'y mean'])
    return df


def groups_info_str(X, y, gps, gp_names):
    """
    Print groups and their sizes.
    """
    ptab = PrettyTable()
    ptab.field_names = ['idx', 'group name', 'n samples', 'fraction', 'y-mean']
    df = groups_info_df(X, y, gps, gp_names)
    n = len(df)

    for i, row in df.iterrows():
        ptab.add_row([i] + row.tolist(), divider=(i == n-2))
    
    ptab.align = 'r'
    ptab.align['group name'] = 'l'
    return ptab.get_string()


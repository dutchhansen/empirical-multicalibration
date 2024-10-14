import numpy as np
from metrics import subgroup_metrics, print_metrics, Logger
from mcb_algorithms.mcb import MulticalibrationPredictor
from relplot import rel_diagram
import matplotlib.pyplot as plt
import os


# One experiment will consist of training some model on a certain split of train/calib,
# and seeing if multicalibration improves subgroup metrics.
class Experiment:
    def __init__(self, dataset, model, calib_frac, calib_train_overlap=0, calib_seed=50):
        '''
        Parameters
            :dataset: Dataset object
            :model: Model object
            :calib_frac: float, fraction of calibration split
            :calib_train_overlap: float, fraction of train set to include in calibration set
            :calib_seed: int, seed for splitting calibration set
        '''
        self.dataset = dataset
        self.model = model
        self.calib_frac = calib_frac
        self.calib_train_overlap = calib_train_overlap
        self.calib_seed = calib_seed
        self.mcb_models = []
        self.logger = None
        self.wandb = False
        
        if (self.calib_frac > 0 or self.calib_train_overlap > 0):
            (
                self.X_train, 
                self.y_train, 
                self.groups_train, 
                self.X_calib, 
                self.y_calib, 
                self.groups_calib
            ) = self.dataset.train_calibration_split(self.calib_frac, 
                                                     train_overlap=calib_train_overlap, 
                                                     seed=calib_seed)

        else:
            self.X_train, self.y_train, self.groups_train = self.dataset.X_train, self.dataset.y_train, self.dataset.groups_train
        
        self.X_test, self.y_test, self.groups_test = self.dataset.X_test, self.dataset.y_test, self.dataset.groups_test
        self.X_val, self.y_val, self.groups_val = self.dataset.X_val, self.dataset.y_val, self.dataset.groups_val

        # Print a nicely formatted table with 
        # train, calibration, validation, and test split sizes
        print(f"\n{'Split':<15}{'Size':<10}{'Fraction of 1s':<15}")
        print(f"{'Train':<15}{len(self.y_train):<10}{np.mean(self.y_train) if len(self.y_train) > 0 else 0:<15.2%}")
        if self.calib_frac > 0:
            print(f"{'Calibration':<15}{len(self.y_calib):<10}{np.mean(self.y_calib):<15.2%}")
        print(f"{'Validation':<15}{len(self.y_val):<10}{np.mean(self.y_val):<15.2%}")
        print(f"{'Test':<15}{len(self.y_test):<10}{np.mean(self.y_test):<15.2%}")
        # include the total length
        print(f"{'Total':<15}{len(self.dataset.y):<10}{np.mean(self.dataset.y):<15.2%}\n")

    def train_model(self):
        print(f"Training {self.model.name} on train split")
        # train model on train split, calibrate on calib split with mcb
        # if calib_frac == 1.0, we cannot train
        if self.calib_frac >= 1.0: return
        self.model.train(self.X_train, self.y_train, self.groups_train, self.X_val, self.y_val, self.groups_val)


    def multicalibrate_multiple(self, config_list):
        '''
        Multicalibrate predictor using multiple algorithms and parameters.

        Params:
            config_list: list of dicts, each containing 'type' and 'params' keys
                        (see configs/constants.py for examples)
        '''
        for alg in config_list:
            alg_type = alg['type']
            params_list = alg['params']
            for params in params_list:
                self.multicalibrate(alg_type=alg_type, params=params)

    def multicalibrate(self, alg_type, params):
        '''
        Multicalibrate predictor using the specified algorithm and parameters.

        Params:
            alg_type: str, the type of algorithm to use for multicalibration
            params: dict, the parameters to use for multicalibration
        '''
        if len(self.X_calib) == 0:
            raise ValueError('No calibration set available for postprocessing.')
        
        print("Multicalibrating model on calib split")
        print(f"Algorithm-type: {alg_type}, Params: {params}")
        # calibrate model on calib ssplit with mcb
        mcbp = MulticalibrationPredictor(alg_type, params)

        # Get probability of positive class
        confs_calib, logits_calib = self.model.predict_proba(self.X_calib, with_logits=True)
        
        # pass in confidence corresponding to correct class
        # mcb algorithms which require logits will use logits_calib
        if alg_type in ['Temp']:
            mcbp.fit(confs=logits_calib, 
                     labels=self.y_calib, 
                     subgroups=self.groups_calib)
        else:
            mcbp.fit(confs=confs_calib,
                    labels=self.y_calib, 
                    subgroups=self.groups_calib)

        self.mcb_models.append([mcbp, alg_type, params])

    def evaluate_val(self, with_rel_diagram=False):
        self.evaluate_model(self.X_val, self.y_val, self.groups_val, 'validation', with_rel_diagram)

    def evaluate_test(self, with_rel_diagram=False):
        self.evaluate_model(self.X_test, self.y_test, self.groups_test, 'test', with_rel_diagram)

    def evaluate_train(self, with_rel_diagram=False):
        self.evaluate_model(self.X_train, self.y_train, self.groups_train, 'train', with_rel_diagram)

    def evaluate_calib(self, with_rel_diagram=False):
        if len(self.X_calib) == 0:
            raise ValueError('No calibration set available for evaluation.')
        
        # warn if calib_train_overlap > 0
        if self.calib_train_overlap > 0:
            print(f"Calibration split includes {self.calib_train_overlap:.2%} of train set")
        
        self.evaluate_model(self.X_calib, self.y_calib, self.groups_calib, 'calibration', with_rel_diagram)
    
    def evaluate_model(self, X, y, groups, dataset_split_name, with_rel_diagram=False):
        # evaluate orig model and mcb model on the given dataset split
        preds = self.model.predict(X)
        (confs, logits) = self.model.predict_proba(X, with_logits=True)
        original_model_metrics_val = subgroup_metrics(groups, y, confs, preds)

        # log metrics
        if self.wandb: self.logger.log("ERM", dataset_split_name, original_model_metrics_val)
        print_metrics(original_model_metrics_val, algorithm=self.model.name, split=dataset_split_name)

        # reliability diagram
        if with_rel_diagram:
            fig, ax = rel_diagram(confs, y)
            dir = f"plots/{self.model.name}/{self.dataset.name}/cf={self.calib_frac}"
            os.makedirs(dir, exist_ok=True)
            fig.savefig(f"{dir}/{dataset_split_name}_ERM.pdf")
            plt.close(fig)

        for (mcbp, alg_type, mcb_params) in self.mcb_models:
            # predict and evaluate for each mcb model we have trained
            # temp scaling needs logits, others need confs
            if alg_type == 'Temp': mcb_confs = mcbp.batch_predict(logits, groups)
            else: mcb_confs = mcbp.batch_predict(confs, groups)
            mcb_preds = np.round(mcb_confs)
            mcb_metrics = subgroup_metrics(groups, y, mcb_confs, mcb_preds)
            
            # log metrics
            if self.wandb: 
                if alg_type == 'HKRR':
                    self.logger.log(f"{alg_type}_L{mcb_params['lambda']}_A{mcb_params['alpha']}", 
                                    dataset_split_name, 
                                    mcb_metrics)
                elif alg_type == 'HJZ':
                    self.logger.log(f"{alg_type}_{mcb_params['algorithm']}" + 
                                    f"_{mcb_params['other_algorithm']}_LR{mcb_params['lr']}" + 
                                    f"_OLR{mcb_params['other_lr']}_B{mcb_params['n_bins']}" + 
                                    f"_I{mcb_params['iterations']}", 
                                    dataset_split_name, 
                                    mcb_metrics)
                elif alg_type == 'Temp':
                    self.logger.log(f"{alg_type}_T{mcb_params['temperature']}", 
                                    dataset_split_name, 
                                    mcb_metrics)
                # for other algs, e.g. calibration methods
                else:
                    self.logger.log(f"{alg_type}", dataset_split_name, mcb_metrics)
            
            # view
            print_metrics(mcb_metrics, algorithm=self.model.name, 
                          postprocess=alg_type, split=dataset_split_name, params=mcb_params)
            
            # reliability diagram
            if with_rel_diagram:
                fig, ax = rel_diagram(mcb_confs, y)
                dir = f"plots/{self.model.name}/{self.dataset.name}/cf={self.calib_frac}"
                os.makedirs(dir, exist_ok=True)
                fig.savefig(f"{dir}/{dataset_split_name}_{alg_type}.pdf")
                plt.close(fig)


    def init_logger(self, config={}, finish=False, project=None, run_name=None):
        """
        Initialize or close the logger object.
        """
        if finish:
            self.logger.finish()
            return

        # init variables
        self.wandb = True
        log_config = {
            **config,
            'dataset': self.dataset.name,
            'model': self.model.name,
            'calib_frac': self.calib_frac,
            'calib_seed': self.calib_seed,
            'groups_str': self.dataset.group_info}
        if not project:
            project = f"{self.dataset.name}_{self.model.name}"

        self.logger = Logger(project, config=log_config, run_name=run_name)

from Dataset import Dataset
from Model import Model
from Experiment import Experiment
from itertools import product
from configs.hyperparameters import get_hyperparameters
from configs.constants import SPLIT_DEFAULT, SEEDS_DEFAULT, CALIB_FRACS_DEFAULT
from configs.constants import SEEDS_REDUCED, CALIB_FRACS_REDUCED
from configs.constants import MCB_DEFAULT
import numpy as np


# helper function for obtaining save directory names
def _save_dir(dataset, model, calib_frac, val_split_seed, include_groups_as_features=False, group_embedding_dim=0):
    return 'models/saved_models/{0}/{1}/calib={2}_val_seed={3}_include_groups_as_features={4}_group_embedding_dim={5}/'.format(
        dataset, model, calib_frac, val_split_seed, include_groups_as_features, group_embedding_dim
        )


def NN_train(model_name, dataset, seeds, include_groups_as_features=False, 
             wandb=True, group_embedding_dim=10):
    '''
    Pretrain model, and evaluate on validation / test sets.
    No multicalibration post-processing.
    '''
    wdb_project = f'{dataset}_{model_name}_eval_pretrain'

    # Calibration fraction set to 0
    cf = 0
    save_scheme = 'best-val-loss'

    hp = get_hyperparameters(model_name, dataset, cf)
    for seed in seeds:
        config = {
            # data
            'dataset': dataset,
            'val_split_seed': seed,
            'split': SPLIT_DEFAULT,
            'calib_frac': cf,
            'include_groups_as_features': include_groups_as_features,
            'use_group_embeddings': group_embedding_dim > 0,
            'embedding_dim': group_embedding_dim,
            # NN
            'model': model_name,
            'save_dir': _save_dir(dataset, 
                                  model_name, 
                                  cf, seed, 
                                  include_groups_as_features=include_groups_as_features,
                                  group_embedding_dim=group_embedding_dim),
            # evaluation
            'val_save_epoch': 0,
            'val_eval_epoch': 1,
            # mcb
            'mcb': [],
            # hyperparameters
            **hp
        }
        config['epochs'] = 5

        dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'], include_groups_as_features=include_groups_as_features)
        if config['include_groups_as_features'] or config['use_group_embeddings']:
            config['num_groups'] = len(dataset_obj.groups)

        # init model
        model = Model(model_name, config=config, SAVE_DIR=config['save_dir'], dataset_obj=dataset_obj, save_scheme=save_scheme)
        experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

        # init logger
        if wandb: experiment.init_logger(config, project=wdb_project)

        # train and postprocess
        experiment.train_model()

        # evaluate
        experiment.evaluate_val()
        experiment.evaluate_test()

        # close logger
        if wandb: experiment.init_logger(finish=True)


if __name__ == "__main__":
    NN_train('DistilBert', 'CivilComments', SEEDS_DEFAULT, include_groups_as_features=False, group_embedding_dim=0)


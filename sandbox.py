from configs.constants import SPLIT_DEFAULT, MCB_DEFAULT
from configs.hyperparameters import get_hyperparameters
from Experiment import Experiment
from Dataset import Dataset
from Model import Model

if __name__ == "__main__":
    '''
    Starter script for running experiments with multicalibration repository.
    '''
    log_to_wandb = False

    # set constants for the experiment
    model_name = 'NaiveBayes'
    dataset = 'ACSIncome'
    mcb_algorithm = 'HKRR'
    mcb_params = MCB_DEFAULT
    calib_frac = 0
    calib_train_overlap = 1.0
    seed = 0

    # one can choose the default collection, or define their own
    groups_collection = 'default'

    # set the save directory and wandb project
    save_dir = 'models/saved_models/{dataset}/{model_name}/calib={calib_frac}_val_seed={seed}/'
    wdb_project = f'{dataset}_project'

    # define config for experiment
    hyp = get_hyperparameters(model_name, dataset, calib_frac)
    config = {
        'model': model_name,                            # model name
        'dataset': dataset,                             # dataset name
        'calib_frac': calib_frac,                       # calibration fraction
        'calib_train_overlap': calib_train_overlap,     # calibration train overlap
        'val_split_seed': seed,                         # seed for validation split
        'split': SPLIT_DEFAULT,                         # default split
        'mcb': [mcb_algorithm],                         # just to keep track of mcb algorithm
        'save_dir': save_dir,                           # save directory
        'val_save_epoch': 0,                            # save model every epoch
        'val_eval_epoch': 1,                            # evaluate model every epoch
        **hyp
    }

    dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'], groups=groups_collection)
    model = Model(model_name, config=config, SAVE_DIR=config['save_dir'])
    experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'], 
                            calib_train_overlap=calib_train_overlap)

    # init logger; this saves metrics to wandb
    if log_to_wandb:
        experiment.init_logger(config, project=wdb_project)

    # train and postprocess
    experiment.train_model()
    if config['calib_frac'] > 0 or config['calib_train_overlap'] > 0:
        experiment.multicalibrate_multiple(mcb_params)

    # evaluate splits
    experiment.evaluate_train()
    experiment.evaluate_calib()
    experiment.evaluate_val()
    # experiment.evaluate_test()

    # close logger; important if you want to open new wandb run
    if log_to_wandb:
        experiment.init_logger(finish=True)
from Dataset import Dataset
from Model import Model
from Experiment import Experiment
from itertools import product
from configs.hyperparameters import get_hyperparameters
from configs.constants import SPLIT_DEFAULT, SEEDS_DEFAULT, CALIB_FRACS_DEFAULT
from configs.constants import SEEDS_REDUCED, CALIB_FRACS_REDUCED
from configs.constants import MCB_DEFAULT


# helper function for obtaining save directory names
def _save_dir(dataset, model, calib_frac, val_split_seed):
    return 'models/saved_models/{0}/{1}/calib={2}_val_seed={3}/'.format(
        dataset, model, calib_frac, val_split_seed
        )


### Experiment primitives ###

def SimpleModel_train_eval(model_name, dataset, calib_fracs, seeds=SEEDS_DEFAULT, wdb_project_name=None, groups_collection='default', wandb=True):
    '''
    Train and evaulate a simple model on several mcb algorithms.
    '''
    if wdb_project_name is None:
        wdb_project = f'{dataset}_{model_name}_eval'
    else:
        wdb_project = wdb_project_name

    for cf in calib_fracs:
        for seed in seeds:
            hp = get_hyperparameters(model_name, dataset, cf)
            config = {
                'model': model_name,
                'dataset': dataset,
                'groups_collection': groups_collection,
                'calib_frac': cf,
                'val_split_seed': seed,
                'split': SPLIT_DEFAULT,
                'mcb': MCB_DEFAULT,
                'save_dir': _save_dir(dataset, model_name, cf, seed),
                **hp
            }

            scale_data = True if 'scale_data' in hp and hp['scale_data'] else False
            dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'], 
                                  scale=scale_data, groups=groups_collection)

            # init model
            model = Model(model_name, config=config, SAVE_DIR=config['save_dir'])
            experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

            # init logger
            if wandb: experiment.init_logger(config, project=wdb_project)

            # train and postprocess
            experiment.train_model()
            if config['calib_frac'] > 0:
                experiment.multicalibrate_multiple(config['mcb'])

            # evaluate
            experiment.evaluate_val()
            experiment.evaluate_test()

            # close logger
            if wandb: experiment.init_logger(finish=True)



def NN_pretrain(model_name, dataset, calib_fracs, seeds, wandb=True):
    '''
    Pretrain model, and evaluate on validation / test sets.
    No multicalibration post-processing.
    '''
    wdb_project = f'{dataset}_{model_name}_eval_pretrain'

    for cf in calib_fracs:
        hp = get_hyperparameters(model_name, dataset, cf)
        for seed in seeds:
            config = {
                # data
                'dataset': dataset,
                'val_split_seed': seed,
                'split': SPLIT_DEFAULT,
                'calib_frac': cf,
                # NN
                'model': model_name,
                'save_dir': _save_dir(dataset, model_name, cf, seed),
                # evaluation
                'val_save_epoch': 0,
                'val_eval_epoch': 1,
                # mcb
                'mcb': [],
                # hyperparameters
                **hp
            }

            dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'])

            # init model
            model = Model(model_name, config=config, SAVE_DIR=config['save_dir'], dataset_obj=dataset_obj)
            experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

            # init logger
            if wandb: experiment.init_logger(config, project=wdb_project)

            # train and postprocess
            experiment.train_model()
            if config['calib_frac'] > 0:
                experiment.multicalibrate_multiple(config['mcb'])

            # evaluate
            experiment.evaluate_val()
            experiment.evaluate_test()

            # close logger
            if wandb: experiment.init_logger(finish=True)


def NN_train_eval(model_name, dataset, calib_fracs, seeds=SEEDS_DEFAULT, wdb_project_name=None, groups_collection='default', wandb=True):
    '''
    Train and evaulate model on validation and test sets.
    Use multicalibration post-processing.
    '''
    if wdb_project_name is None:
        wdb_project = f'{dataset}_{model_name}_eval'
    else:
        wdb_project = wdb_project_name

    for cf in calib_fracs:
        hp = get_hyperparameters(model_name, dataset, cf)
        for seed in seeds:
            config = {
                # data
                'dataset': dataset,
                'groups_collection': 'default',
                'val_split_seed': seed,
                'split': SPLIT_DEFAULT,
                'calib_frac': cf,
                # NN
                'model': model_name,
                'save_dir': _save_dir(dataset, model_name, cf, seed),
                # evaluation
                'val_save_epoch': 0,
                'val_eval_epoch': 1,
                # mcb
                'mcb': MCB_DEFAULT,
                # hyperparameters
                **hp
            }

            dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'], groups=groups_collection)

            # init model
            model = Model(model_name, config=config, dataset_obj=dataset_obj, SAVE_DIR=config['save_dir'])
            experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

            # init logger
            run_name = f'cf={cf}_seed={seed}_epoch={hp["epochs"]-1}'
            if wandb: experiment.init_logger(config, project=wdb_project, run_name=run_name)

            # train and postprocess
            experiment.train_model()
            if config['calib_frac'] > 0:
                experiment.multicalibrate_multiple(config['mcb'])

            # evaluate
            experiment.evaluate_val()
            experiment.evaluate_test()

            # close logger
            if wandb: experiment.init_logger(finish=True)


def NN_eval(model_name, dataset, calib_fracs, seeds, eval_epochs=None, no_mcb=False, wandb=True):
    '''
    No training. Evaluate a pretrained model with several mcb algorithms.
    We use this function to evaluate larger models, such as DistilBERT and ViT
    '''
    wdb_project = f'{dataset}_{model_name}_eval'

    for cf in calib_fracs:
        hp = get_hyperparameters(model_name, dataset, cf)
        num_epochs = hp['epochs']
        epochs = [num_epochs - 1]
        if eval_epochs is not None:
            epochs = eval_epochs

        # if image resnet, can only evaluate at last epoch
        if model_name in ['ImageResNet', 'MLP']:
            err_msg = f'{model_name} can only evaluate at last epoch; all-epoch saving is not supported.'
            assert epochs == [num_epochs - 1], err_msg

        for seed in seeds:
            for e in epochs:
                print(f'********** {dataset} {model_name} cf={cf} seed={seed} epoch={e} **********')
                config = {
                    # evaluation
                    'eval_epoch': e,
                    # data
                    'dataset': dataset,
                    'val_split_seed': seed,
                    'split': SPLIT_DEFAULT,
                    'calib_frac': cf,
                    # NN
                    'model': model_name,
                    'save_dir': _save_dir(dataset, model_name, cf, seed),
                    # evaluation
                    'val_save_epoch': 0,
                    'val_eval_epoch': 1,
                    # mcb
                    'mcb': [] if no_mcb else MCB_DEFAULT,
                    # hyperparameters
                    **hp
                }

                # create duplicate config, and alter only batch size
                # this allows for evaluation with less memory
                config_low_bs = config.copy()
                config_low_bs['batch_size'] = 8
                dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'])

                # init model
                model = Model(model_name, config=config_low_bs, SAVE_DIR=config['save_dir'],
                              dataset_obj=dataset_obj, from_saved=True, saved_epoch=e)
                experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

                # init logger
                run_name = f'cf={cf}_seed={seed}_epoch={e}'
                if wandb: experiment.init_logger(config, project=wdb_project, run_name=run_name)

                # train and postprocess
                if config['calib_frac'] > 0:
                    experiment.multicalibrate_multiple(config['mcb'])

                # evaluate
                experiment.evaluate_val()
                experiment.evaluate_test()

                # close logger
                if wandb: experiment.init_logger(finish=True)


def data_reuse_experiment(model_name, dataset, seed, wandb=True):
    # set constants for the experiment
    mcb_params = MCB_DEFAULT
    calib_frac = 0
    calib_train_overlap = 1.0
    groups_collection = 'default'

    # set the save directory and wandb project
    save_dir = 'models/saved_models/{dataset}/{model_name}/calib={calib_frac}_val_seed={seed}/'
    wdb_project = f'{dataset}_{model_name}_reuse-train-data'

    # define config for experiment
    hyp = get_hyperparameters(model_name, dataset, calib_frac)
    config = {
        'model': model_name,                            # model name
        'dataset': dataset,                             # dataset name
        'group_collection': groups_collection,          # group collection
        'calib_frac': calib_frac,                       # calibration fraction
        'calib_train_overlap': calib_train_overlap,     # calibration train overlap
        'val_split_seed': seed,                         # seed for validation split
        'split': SPLIT_DEFAULT,                         # default split
        'mcb': MCB_DEFAULT,                             # just to keep track of mcb algorithm
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
    if wandb: experiment.init_logger(config, project=wdb_project)
    
    # train and postprocess
    experiment.train_model()
    if config['calib_frac'] > 0 or config['calib_train_overlap'] > 0:
        experiment.multicalibrate_multiple(mcb_params)

    # evaluate splits
    # experiment.evaluate_train()
    # experiment.evaluate_calib()
    experiment.evaluate_val()
    experiment.evaluate_test()

    # close logger
    if wandb: experiment.init_logger(finish=True)


### Multi-experiment functions ###

def eval_all_SimpleModel(datasets, calib_fracs, models, seeds=SEEDS_DEFAULT):
    '''
    Helper function for evaluating SimpleModels with optimal 
    hyperparameters on all datasets.
    '''
    for dataset in datasets:
        for model in models:
            print(f'********** {dataset} {model} **********')
            SimpleModel_train_eval(model, dataset, calib_fracs, seeds)


def eval_all_MLP():
    '''
    Helper function for evaluating MLPs with optimal 
    hyperparameters on all datasets.
    '''
    for dataset in ['ACSIncome', 'BankMarketing', 'CreditDefault', 'MEPS', 'HMDA']:
        NN_train_eval('MLP', dataset, CALIB_FRACS_DEFAULT)


def data_reuse_experiments_all():
    models = [
        'SVM', 
        'LogisticRegression', 
        'NaiveBayes', 
        'DecisionTree', 
        'RandomForest', 
        'MLP'
        ]
    
    datasets = ['ACSIncome', 'BankMarketing', 'CreditDefault', 'MEPS', 'HMDA']
    seeds = SEEDS_DEFAULT

    # create all combinations of datasets, models, and seeds
    combs = product(datasets, models, seeds)
    for dataset, model, seed in combs:
        print(f'********** {dataset} {model} seed={seed} **********')
        data_reuse_experiment(model, dataset, seed)


if __name__ == '__main__':
    '''
    One may call experiments from here. All experiments are logged to wandb, 
    and each project has a name of the form '{dataset}_{model}_eval'. This is 
    to differentiate from the projects titled '{dataset}_{model}_search', 
    which are used for hyperparameter search.
    '''

    # example usage #1:
    # SimpleModel_train_eval('SVM', 'ACSIncome', [0.4], seeds=[55, 45])
    
    # example usage #2:
    # NN_train_eval('MLP', 'HMDA',
    #                  [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                  seeds=[15, 25, 35, 45, 55],
    #                  wdb_project_name='HMDA_MLP_alternate_eval', # custom project name
    #                  groups_collection='alternate',
    #                  wandb=True)

    # example usage #3:
    # data_reuse_experiments_all()

    # example usage #4:
    # for dataset in ['ACSIncome']:
    #     for model in ['NaiveBayes']:
    #         SimpleModel_train_eval(model, dataset, 
    #                             calib_fracs=[0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             seeds=[15, 25, 35, 45, 55], 
    #                             wdb_project_name=f'{dataset}_{model}_alternate_eval', 
    #                             groups_collection='alternate', 
    #                             wandb=True)
    

    pass


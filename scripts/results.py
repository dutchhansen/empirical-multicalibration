import wandb
import numpy as np
import pandas as pd
import prettytable as ptab
from collections import defaultdict
import os


API = wandb.Api(timeout=120)
DEFAULT_METRIC_SUFFIXES = ['acc/agg', 'ECE/agg', 'ECE/max', 'smECE/agg', 'smECE/max']


def load_runs(project, filters={}):
    '''
    Return list of runs for given project and filters.
    '''
    runs_list = API.runs(path=project, filters={
        'state': 'finished',
        **filters
        })
    return runs_list


def get_metric_name_map(algorithm_type, params, metric_suffixes, split):
    """
    Get mapping from metrics suffixes to their logged names on wandb.
    Supported algorithms: ERM, HKRR, HJZ
    """
    if algorithm_type == 'ERM':
        metric_name_map = {
            suffix: f"ERM/{split}/{suffix}" 
            for suffix in metric_suffixes
            }
    elif algorithm_type == 'HKRR':
        metric_name_map = {
            suffix: f"HKRR_L{params['lambda']}_A{params['alpha']}/{split}/{suffix}" 
            for suffix in metric_suffixes}
    elif algorithm_type == 'HJZ':
        metric_name_map = {
            suffix: f"HJZ_{params['algorithm']}"+
                    f"_{params['other_algorithm']}" + 
                    f"_LR{params['lr']}" + 
                    f"_OLR{params['other_lr']}" + 
                    f"_B{params['n_bins']}" + 
                    f"_I{params['iterations']}/{split}/{suffix}"
            for suffix in metric_suffixes
        }
    elif algorithm_type == 'Temp':
        metric_name_map = {
            suffix: f"Temp_T{params['temperature']}/{split}/{suffix}"
            for suffix in metric_suffixes
        }
    else:
        metric_name_map = {
            suffix: f"{algorithm_type}/{split}/{suffix}" 
            for suffix in metric_suffixes
        }
    
    return metric_name_map


'''
Here, we build configuration objects necessary to create tables.
This differs from the configuration objects in constants.py, as we
we enable one to specify a range of parameters for each calibration fraction.
'''

# HJZ parameters
HJZ_ALGS = ["Hedge", "MLProd", "OnlineGradientDescent", "OptimisticHedge"]
HJZ_lrs_reduced = [0.9, 0.95]
HJZ_other_lrs_reduced = [0.9, .95, 0.98]
HJZ_lr_pairs_reduced = [(a, b) for a in HJZ_lrs_reduced for b in HJZ_other_lrs_reduced]

# build HKRR algorithms config
HKRR_PARAMS = [
            {'lambda': 0.1, 'alpha': 0.1},
            {'lambda': 0.1, 'alpha': 0.05},
            {'lambda': 0.1, 'alpha': 0.025},
            {'lambda': 0.1, 'alpha': 0.0125},
        ]

# build HJZ algorithms configs
HJZ_PARAMS_1 = [{
            'algorithm': alg,
            'other_algorithm': 'None',
            'lr': a,
            'other_lr': 0,
            'n_bins': 10,
            'iterations': 30,
        } for alg in HJZ_ALGS for a in HJZ_lrs_reduced]
HJZ_PARAMS_2 = [{
            'algorithm': alg,
            'other_algorithm': alg,
            'lr': a,
            'other_lr': b,
            'n_bins': 10,
            'iterations': 30,
        } for alg in ["Hedge", "OptimisticHedge"] for a, b in HJZ_lr_pairs_reduced]

# calibration methods
PLATT_PARAMS = [{}]
ISOTONIC_PARAMS = [{}]
TEMP_PARAMS = [{'temperature': None, 'optimized': True}] + [
            {'temperature': round(t, 1), 'optimized': False} for t in np.linspace(0.2, 4, 20)
        ]

# collect configs
DEFAULT_MCB_ALGS = {
    cf: {
        'HKRR': HKRR_PARAMS,
        'HJZ': HJZ_PARAMS_1 + HJZ_PARAMS_2,
        'Platt': PLATT_PARAMS,
        'Isotonic': ISOTONIC_PARAMS,
        'Temp': TEMP_PARAMS
    } for cf in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]}
DEFAULT_MCB_ALGS[0] = {'ERM': [{}]}


def metrics_from_df(df, row_idx, algorithm_type, params, metric_suffixes=DEFAULT_METRIC_SUFFIXES, split='validation'):
    metric_name_map = get_metric_name_map(algorithm_type, params, metric_suffixes, split)
    metrics_map = {}
    res = {}

    # search through keys in rh
    for metric in metric_name_map:
        metric_name = metric_name_map[metric]
        metrics_map[metric] = df.iloc[row_idx][metric_name]

    # fill metrics map
    res = {}
    for metric in metric_name_map:
        if metric in metrics_map:
            res[metric] = metrics_map[metric]
        else:
            raise ValueError(f"current run: {algorithm_type, params, split}; " + 
                             f"{metric} not in among metrics. Requested metric " + 
                             f"does not exist.")

    return res


def experiments_table(model_name, 
                      dataset_name,
                      split='validation',
                      alg_config=DEFAULT_MCB_ALGS, 
                      metric_suffixes=DEFAULT_METRIC_SUFFIXES,
                      return_param_dict=False,
                      results_dir='results'):
    """
    Generates pandas dataframe of experimental data associated with the input model 
    and dataset, and with the alg_config collection of mcb algorithms.
    Draws data from csv saved as results_dir/{model_name}/{dataset_name}_{model_name}.csv.

    Parameters:
        :model: str, name of model
        :dataset: str, name of dataset
        :split: str, name of split
        :alg_config: dict, mapping calib_frac -> post-processing alg -> list of parameters
        :metric_suffixes: list, list of metric suffixes to record

    Returns: pandas dataframe, table of experimental data
    """
    project = f'{dataset_name}_{model_name}'
    path = os.path.join(results_dir, model_name, f'{dataset_name}_{model_name}.csv')
    results = pd.read_csv(path)

    # idx_to_param: calib_frac -> alg_type -> param_idx -> params
    idx_to_param = defaultdict(lambda: defaultdict(dict))
    # metrics: calib_frac -> alg_type -> param_idx -> list of metrics maps
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row_idx in range(len(results)):
        cf = results.iloc[row_idx]['calib_frac']
        for alg_type, params_list in alg_config[cf].items():
            for param_idx, params in enumerate(params_list):

                # track idx <-> param correspondence
                idx_to_param[cf][alg_type][param_idx] = params
                mm = metrics_from_df(results, row_idx, alg_type, params, metric_suffixes, split)
                metrics[cf][alg_type][param_idx].append(mm)
    
    # print table
    t = ptab.PrettyTable()
    columns = ['calib frac', 'alg type', 'param id'] + metric_suffixes
    data, data_sd = [], []
    
    # calculate mean and sd for each metric in category
    for cf in metrics:
        for alg_type in metrics[cf]:
            for param_idx in metrics[cf][alg_type]:

                # calculate mean and sd for each metric in category
                mean, sd = {}, {}
                for m in metric_suffixes:
                    # mean
                    s = sum([run[m] for run in metrics[cf][alg_type][param_idx]])
                    n = len(metrics[cf][alg_type][param_idx])
                    mean[m] = s / n

                    # standard deviation
                    sq_diff = sum([(run[m] - mean[m])**2 for run in metrics[cf][alg_type][param_idx]])
                    sd[m] = (sq_diff / n) ** 0.5
            
                # add row to both mean and sd dataframes
                data.append([cf, alg_type, param_idx] + [mean[m] for m in metric_suffixes])
                data_sd.append([cf, alg_type, param_idx] + [sd[m] for m in metric_suffixes])

    # convert to pandas dataframe
    df = pd.DataFrame(data, columns=columns)
    sd_df = pd.DataFrame(data_sd, columns=columns)

    # Get rid of spaces in column names, which sometimes causes issues
    df.columns = [c.replace(' ', '_') for c in df.columns]
    sd_df.columns = [c.replace(' ', '_') for c in sd_df.columns]

    if return_param_dict:
        return df, sd_df, idx_to_param
    
    return df, sd_df


def download_metrics(project, save_path):
    runs = load_runs(project)
    rows = []
    for run in runs:
        config = run.config
        metrics = run.summary_metrics
        row = {**config, **metrics}
        rows.append(row)
    runs_df = pd.DataFrame(rows)

    # delete old file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    runs_df.to_csv(save_path)
    print(f"File size of saved csv: {os.path.getsize(save_path) / 1024 / 1024} MB")
    print(f"rows: {runs_df.shape[0]}, columns: {runs_df.shape[1]}")


def download_experiment(model, dataset, specific_model_name=None):
    # loading csvs with the complete wandb history
    # takes much longer, so we save them in a separate directory
    # wandb project
    project = f'{dataset}_{model}_eval'

    # name under which we save
    model_name = model
    if specific_model_name is not None:
        model_name = specific_model_name
    
    save_dir = 'results_complete'
    directory = f'{save_dir}/{model_name}'
    path = f'{save_dir}/{model_name}/{dataset}_{model_name}.csv'
    
    # check that directory exists
    os.makedirs(directory, exist_ok=True)
    download_metrics(project, path)


def download_tabular_experiments():
    '''
    Download all experiments for all models and datasets.
    '''
    for dataset in ['ACSIncome', 'BankMarketing', 'CreditDefault', 'HMDA', 'MEPS']:
        for model in ['MLP', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'NaiveBayes']: #  SVM left out for now
            download_experiment(model, dataset)
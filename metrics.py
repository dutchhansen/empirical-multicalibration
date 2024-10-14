import numpy as np
from utils import ConsoleColor
from relplot import smECE as _smECE
from prettytable import PrettyTable as pt
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb


def binnedECE(f, y):
    '''Expected Calibration Error (binned)'''
    def _binning(f, y, bin_size=0.1, shift=0):
        bi = (f + shift) / bin_size
        bi = bi.astype(int)
        r = f - y
        bins_cnt = int((shift + 1) / bin_size) + 1
        bins = np.zeros(bins_cnt)
        np.add.at(bins, bi, r)
        return bins
    
    def _binnedECE(f, y, nbins=10):
        bins = _binning(f, y, bin_size=1.0 / nbins, shift=0)
        return np.abs(bins).sum() / len(f)
    
    return _binnedECE(f, y)


def smECE(f, y):
    '''Expected Calibration Error (smoothed)'''
    return _smECE(f, y)


def subgroup_metrics(subgroups, targets, positive_class_confs, preds):
    '''return dictionary of groupwise calibration metrics'''
    subgroup_metrics = {}

    for i, group in enumerate(subgroups):
        subgroup_frac = len(group) / len(positive_class_confs)

        # check for empty subgroups
        if len(group) <= 1:
            subgroup_metrics[i] = {
                "size": round(subgroup_frac,4), 
                "acc": 'NA',
                "ECE": 'NA',
                "smECE": 'NA',
            }
            continue

        subgroup_confs = positive_class_confs[group]
        subgroup_preds = preds[group]
        subgroup_targets = targets[group]

        # deterministic metrics
        group_acc = len(np.where(subgroup_preds == subgroup_targets)[0]) / len(group)
        ece = binnedECE(subgroup_confs, subgroup_targets)
        smece = smECE(subgroup_confs, subgroup_targets)

        subgroup_metrics[i] = {
            "size": round(subgroup_frac,4), 
            "acc": round(group_acc,4),
            "ECE": round(ece,4),
            "smECE": round(smece, 4)}

    # get aggregate metrics
    agg_metrics = {
        "size": 1.0,
        "acc": round(len(np.where(preds == targets)[0]) / len(targets), 4),
        "ECE": round(binnedECE(positive_class_confs, targets), 4),
        "smECE": round(smECE(positive_class_confs, targets), 4)
    }

    # add mean subgroup metrics
    sg_mean = {
        "size": round(np.mean([subgroup_metrics[i]['size'] for i in subgroup_metrics]), 4),
        "acc": round(np.mean([subgroup_metrics[i]['acc'] for i in subgroup_metrics if subgroup_metrics[i]['acc'] != 'NA']), 4),
        "ECE": round(np.mean([subgroup_metrics[i]['ECE'] for i in subgroup_metrics if subgroup_metrics[i]['ECE'] != 'NA']), 4),
        "smECE": round(np.mean([subgroup_metrics[i]['smECE'] for i in subgroup_metrics if subgroup_metrics[i]['smECE'] != 'NA']), 4)
    }

    # add subgroup max
    sg_max = {
        "size": round(np.max([subgroup_metrics[i]['size'] for i in subgroup_metrics]), 4),
        "acc": round(np.max([subgroup_metrics[i]['acc'] for i in subgroup_metrics if subgroup_metrics[i]['acc'] != 'NA']), 4),
        "ECE": round(np.max([subgroup_metrics[i]['ECE'] for i in subgroup_metrics if subgroup_metrics[i]['ECE'] != 'NA']), 4),
        "smECE": round(np.max([subgroup_metrics[i]['smECE'] for i in subgroup_metrics if subgroup_metrics[i]['smECE'] != 'NA']), 4)
    }

    # add subgroup min
    sg_min = {
        "size": round(np.min([subgroup_metrics[i]['size'] for i in subgroup_metrics]), 4),
        "acc": round(np.min([subgroup_metrics[i]['acc'] for i in subgroup_metrics if subgroup_metrics[i]['acc'] != 'NA']), 4),
        "ECE": round(np.min([subgroup_metrics[i]['ECE'] for i in subgroup_metrics if subgroup_metrics[i]['ECE'] != 'NA']), 4),
        "smECE": round(np.min([subgroup_metrics[i]['smECE'] for i in subgroup_metrics if subgroup_metrics[i]['smECE'] != 'NA']), 4)
    }

    # subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['max'] = sg_max
    subgroup_metrics['min'] = sg_min
    subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['agg'] = agg_metrics

    return subgroup_metrics


def print_metrics(metrics_dict, algorithm="-", postprocess="", split="-", params="-"):
    table = pt()
    table.field_names = ["Group", "size", 'Acc', 'ECE', 'smECE']
    algorithm_name = algorithm if postprocess == "" else f"{algorithm} + {ConsoleColor.BLUE}{postprocess}{ConsoleColor.END}"
    table.title = f"{algorithm_name} : {ConsoleColor.CYAN}{split}{ConsoleColor.END} : {params}"
    # look among rows (0, ..., n_rows-1) for worst metric value
    n_rows = len(metrics_dict) - 4
    n_metrics = 3

    # place values in temp table
    t = [[group, metrics['size'], metrics['acc'], metrics['ECE'], metrics['smECE']]
        for group, metrics in metrics_dict.items()]

    # track criteria for worst metric across groups
    # metrics assumed to start at column 2 of table
    field_idxs = {table.field_names[i]: i for i in range(2, n_metrics + 2)}
    worst_criteria = {"Acc": "lt", "ECE": "gt", "smECE": "gt"}
    worst_idxs = {"Acc": 0, "ECE": 0, "smECE": 0}

    # find worst metric across groups
    for i in range(n_rows):
        for j in range(2, n_metrics + 2):
            metric, val = table.field_names[j], t[i][j]
            # if metric is worse than current worst, update worst
            if worst_criteria[metric] == "lt" and val < t[worst_idxs[metric]][j]:
                worst_idxs[metric] = i
            elif worst_criteria[metric] == "gt" and val > t[worst_idxs[metric]][j]:
                worst_idxs[metric] = i
            
    # color worst metric with red, and reformat numbers
    for i in range(n_rows):
        for j in range(2, n_metrics + 2):
            metric, val = table.field_names[j], t[i][j]
            if i == worst_idxs[metric]:
                t[i][j] = f"{ConsoleColor.BOLD}{ConsoleColor.RED}{val:.4f}{ConsoleColor.END}"
            else:
                t[i][j] = f"{val:.4f}"
    for c, idx in worst_idxs.items():
        t[idx][field_idxs[c]] = f"{ConsoleColor.BOLD}{ConsoleColor.RED}{t[idx][field_idxs[c]]}{ConsoleColor.END}"

    # # embolden max and agg rows
    # for r in t:
    #     if r[0] in ['agg']:
    #         for i in range(len(r)):
    #             r[i] = f"{ConsoleColor.BOLD}{r[i]}{ConsoleColor.END}"

    for i, r in enumerate(t):
        table.add_row(r, divider=(i == len(t) - 5))

    table.align = "l"
    table.align["Group"] = "r"
    print(table)


class Logger:
    def __init__(self, project, config, run_name=None):
        self.project = project
        self.run = run_name
        self.config = config
        wandb.init(project=project, name=run_name, config=config)

    def log(self, model, split, group_metrics):
        """
        Log group metrics to wandb.
        Parameters:
            model (str): The stage of the experiment, e.g. 'ERM', 'MCB'
            group_metrics (dict): The groupwise calibration metrics {group_index: {metric: value}}
        """
        log_dict = defaultdict(dict)
        for gp in group_metrics:
            for metric, value in group_metrics[gp].items():
                log_dict[f"{model}/{split}/{metric}/{gp}"] = value

        wandb.log(log_dict)

    def log_plot(self, name, plot):
        wandb.log({name: plot})

    def finish(self):
        wandb.finish()


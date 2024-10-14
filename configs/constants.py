import numpy as np

'''
Here, we store a number of constant values used throughout experiments.
Primarily, we document default values for the hyperparameters of the MCB algorithms.
'''

# default splits and seeds
SPLIT_DEFAULT = {'train': 0.6, 'val': 0.2, 'test': 0.2}
SEEDS_DEFAULT = [15, 25, 35, 45, 55]
SEEDS_REDUCED = [15, 25, 35]
CALIB_FRACS_DEFAULT = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CALIB_FRACS_REDUCED = [0, 0.2, 0.4]
LARGER_MLPS = [
    [128, 256, 128],
    [128, 'BN', 256, 'BN', 128]
]

# default hyperparameters for HKRR
HKRR_DEFAULT = [
    {
        'type': 'HKRR',
        'params': [{
            'lambda': 0.1,
            'alpha': 0.1,
        }, {
            'lambda': 0.1,
            'alpha': 0.05,
        }, {
            'lambda': 0.1,
            'alpha': 0.025,
        }, {
            'lambda': 0.1,
            'alpha': 0.0125,
        }]
    }, 
]

# we use a reduced version of the sweep in Haghtalab et al. 2021
HJZ_ALGS = ["Hedge", "MLProd", "OnlineGradientDescent", "OptimisticHedge"]
HJZ_lrs = [0.8, 0.85, 0.9, 0.95]
HJZ_other_lrs = [0.9, 0.95, 0.98, 0.99]
HJZ_lr_pairs = [(a, b) for a in HJZ_lrs for b in HJZ_other_lrs]

HJZ_lrs_reduced = [0.9, 0.95]
HJZ_other_lrs_reduced = [0.9, .95, 0.98]
HJZ_lr_pairs_reduced = [(a, b) for a in HJZ_lrs_reduced for b in HJZ_other_lrs_reduced]

HJZ_DEFAULT = [
    {
        'type': 'HJZ',
        'params': [{
            'algorithm': alg,
            'other_algorithm': 'None',
            'lr': a,
            'other_lr': 0,
            'n_bins': 10,
            'iterations': 30,
        } for alg in HJZ_ALGS for a in HJZ_lrs_reduced]
    },
    {
        'type': 'HJZ',
        'params': [{
            'algorithm': alg,
            'other_algorithm': alg,
            'lr': a,
            'other_lr': b,
            'n_bins': 10,
            'iterations': 30,
        } for alg in ["Hedge", "OptimisticHedge"] for a, b in HJZ_lr_pairs_reduced]
    }
]

# standard calibration algorithms
CALIB_ALGS_DEFAULT = [
    {
        'type': 'Platt',
        'params': [{}]
    },
    {
        'type': 'Isotonic',
        'params': [{}]
    },
    {
        'type': 'Temp',
        'params': [{'temperature': None, 'optimized': True}] + 
                   [{'temperature': round(t, 1), 'optimized': False} 
                    for t in np.linspace(0.2, 4, 20)]
    }
]

# collect all mcb algorithsm
MCB_DEFAULT = HKRR_DEFAULT + HJZ_DEFAULT + CALIB_ALGS_DEFAULT
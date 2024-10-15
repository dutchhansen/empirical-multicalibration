# Multicalibration Post-Processing

This repository contains the official implementation of all experiments in ["When is Multicalibration Post-Processing Necessary?"](https://arxiv.org/abs/2406.06487v1) We conduct the first
comprehensive empirical study of multicalibration post-processing,
across a broad family of tabular, image, and language datasets, for models spanning
from simple decision trees to 90 million parameter fine-tuned LLMs.

Included are several standalone tools for the study of multicalibration. 

## Requirements

Original experiments were run in a Conda environment using Python 3.11. Before proceeding, we recommend updating pip. To install necessary packages:

```setup
pip install -r requirements.txt
```

## Reproducing Results

All experimental results are provided in the `results` directory. To create figures from these results, run the following command in the root directory:

```bash
python scripts/generate_figures.py
```

To run an experiment, run one of the functions available in `experiments.py`. Given a model, dataset, list of calibration fractions, and list of seeds for the validation split, these functions will pretrain, train, or evaluate (depending on the chosen script) over the specified calibration fractions and split seeds. To specify the model hyperparameters on each dataset and calibration fraction, one may edit the `hyperparameters` dictionary in `configs/hyperparameters.py`, though it currently contains the hyperparameters used to obtain our results.

Once models have been trained, post-processed, and evaluated, one may reproduce the figures for these new runs. To first download the results from wandb, run the script provided in `download_results.py`. This will download the entire collection of wandb runs as csvs, which will be stored in the `results` directory. Once this information is saved, one may freely generate figures with the scripts cited above.

This process will reproduce our primary results. To reproduce results from our "alternate" groups and data-reuse experiments, one use the following more general intstructions.

## Using This Repository

Training a model and applying a post-processing algorithm is straightforward. Some starter code is provided in `sandbox.py`, though we walk through this code more explicitly here. Consider the following example, which uses hyperparameters from our paper.

First define a multicalibration algorithm. To see available algorithms, examine the code in the `mcb_algorithms` subdirectory. A number of the parameters we use are hard-coded in `configs/constants.py`. Alternatively, one may define their own parameters as demonstrated below.

```python
mcb_algorithm = 'HKRR'
mcb_params = {
    'lambda': 0.1,
    'alpha': 0.025,
}
```

From here, running each postprocess follows from calling `experiment.multicalibrate()` with the desired algorithm and parameters. Due to the computational cost of these algorithms, it is sometimes easier to run several of them after training a single base predictor. To do this, one may use `experiment.multicalibrate_multiple()`; for usage instructions, see the docstring in `Experiment.py`.

```python
from configs.constants import SPLIT_DEFAULT
from configs.hyperparameters import get_hyperparameters
from Experiment import Experiment
from Dataset import Dataset
from Model import Model

# set constants for the experiment
dataset = 'ACSIncome'
model_name = 'MLP'
mcb_algorithm = 'HKRR'
mcb_params = {'lambda': 0.1, 'alpha': 0.025}
calib_frac = 0.4
seed = 0

# set the save directory for model and wandb project name
save_dir = 'models/saved_models/{dataset}/{model_name}/calib={calib_frac}_val_seed={seed}/'
wdb_project = f'{dataset}_sandbox_project'

# define config for experiment
hyp = get_hyperparameters(model_name, dataset, calib_frac)
config = {
    'model': model_name,        # model name
    'dataset': dataset,         # dataset name
    'calib_frac': calib_frac,   # calibration fraction
    'val_split_seed': seed,     # seed for validation split
    'split': SPLIT_DEFAULT,     # default split
    'mcb': [mcb_algorithm],     # just to keep track of mcb algorithm
    'save_dir': save_dir,       # save directory
    'val_save_epoch': 0,        # epoch after which we save model
    'val_eval_epoch': 1,        # eval model when epoch % val_eval_epoch == 0
    **hyp
}

dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'])
model = Model(model_name, config=config, SAVE_DIR=config['save_dir'])
experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

# init logger; this saves metrics to wandb
experiment.init_logger(config, project=wdb_project)

# train and postprocess
experiment.train_model()
if config['calib_frac'] > 0:
    experiment.multicalibrate(mcb_algorithm, mcb_params)

# evaluate splits
experiment.evaluate_val()
experiment.evaluate_test()

# close logger
experiment.init_logger(finish=True)
```

Using the `get_hyperparameters()` function, this example uses pre-defined hyperparameters for the base predictor. It is possible to specify custom hyperparameters. To do this, change the appropriate key-value pairs in the `hyp` dictionary; to see available keys and values, examine `configs/hyperparameters.py`.


## Citation

If you found our code to be useful, please consider citing our work.
```
@inproceedings{hansen2024multicalibration,
  title={When is Multicalibration Post-Processing Necessary?},
  author={Hansen, Dutch and Devic, Siddartha and Nakkiran, Preetum and Sharan, Vatsal},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgements

Our multicalibration implementations are based on code written by Eric Zhao [[1]](https://github.com/ericzhao28/multicalibration) and Sana Tonekaboni [[2]](https://github.com/sanatonek/fairness-and-callibration/tree/master). The [WILDS Benchmark](https://github.com/p-lambda/wilds) also provided inspiration for the design of this repository. We thank Bhavya Vasudeva for helpful discussions.

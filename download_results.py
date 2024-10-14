from scripts.results import load_runs
import pandas as pd
import os

def download_metrics(project, save_path):
    '''
    Downloads metrics from entire wandb project, 
    and saves them to a csv file.
    '''
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
    '''
    Downloads metrics from wandb project, 
    and saves them to a csv file. We log all final runs
    to "_eval" projects, and we make this assumption here.
    '''
    # wandb project
    project = f'{dataset}_{model}_eval'

    # name under which we save
    model_name = model
    if specific_model_name is not None:
        model_name = specific_model_name
    
    save_dir = 'results'
    directory = f'{save_dir}/{model_name}'
    path = f'{save_dir}/{model_name}/{dataset}_{model_name}.csv'
    
    # check that directory exists
    os.makedirs(directory, exist_ok=True)
    download_metrics(project, path)


def download_tabular_experiments():
    '''Download all experiments for all models and datasets.'''
    for dataset in ['ACSIncome', 'BankMarketing', 'CreditDefault', 'HMDA', 'MEPS']:
        for model in ['MLP', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'NaiveBayes', 'SVM']:
            download_experiment(model, dataset)

if __name__ == '__main__':
    '''
    Once you have run all of the desired experiments, you can download wandb
    results to csv using functions from scripts/results.py.

    WARNING: This script will not work properly if some of the
    experiments have not yet been run.
    '''
    download_tabular_experiments()
    download_experiment('ViT', 'CelebA')
    download_experiment('ViT', 'Camelyon17')
    download_experiment('ImageResNet', 'CelebA', 'ResNet-50')
    download_experiment('ImageResNet', 'Camelyon17', 'DenseNet-121')
    download_experiment('LanguageResNet', 'AmazonPolarity', 'ResNet-56')
    download_experiment('DistilBert', 'CivilComments', 'DistilBERT')

    pass
from plots import plot_all_mcb_algs, increasing_CF_helps_mcb_algs, increasing_CF_hurts_acc
from tables import generate_table_tabular_datasets, data_reuse_table_tabular_datasets


def tabular(groups_collection='default'):
    '''
    Generate tables and figures for tabular datasets.
    Provides option for alternate groups.
    '''
    datasets = ['ACSIncome', 'BankMarketing', 'CreditDefault', 'HMDA', 'MEPS']
    models = ['MLP', 'RandomForest', 'SVM', 'LogisticRegression', 'DecisionTree', 'NaiveBayes']
    results_dir = 'results'
    save_dir = 'figures/'

    if groups_collection == 'alternate':
        results_dir = 'results/alternate_groups'
        save_dir = 'figures/alternate_groups/'

    print(f'Generating tables and figures for tabular datasets with {groups_collection} groups.')

    for dataset in datasets:
        for model in models:
            # generate figures
            plot_all_mcb_algs(dataset, model, save_dir, select_on_attribute=None, results_dir=results_dir)
            increasing_CF_helps_mcb_algs(dataset, model, save_dir, select_on_attribute='smECE/max', results_dir=results_dir)
            increasing_CF_hurts_acc(dataset, model, save_dir, select_on_attribute='smECE/max', results_dir=results_dir)

        # generate tables
        table_dir = 'figures/alternate_groups/tables/'
        generate_table_tabular_datasets(models, dataset, table_dir, results_dir=results_dir)


def non_tabular():
    '''
    Generate tables and figures for non-tabular datasets.
    '''
    # Generate plots for vision experiments
    print('Generating plots for vision datasets.')
    plot_all_mcb_algs('Camelyon17', 'DenseNet-121', 'figures/', select_on_attribute=None)
    plot_all_mcb_algs('Camelyon17', 'ViT', 'figures/', select_on_attribute=None)
    plot_all_mcb_algs('CelebA', 'ViT', 'figures/', select_on_attribute=None)
    plot_all_mcb_algs('CelebA', 'ResNet-50', 'figures/', select_on_attribute=None)

    # Generate plots for language experiments
    print('Generating plots for language datasets.')
    plot_all_mcb_algs('CivilComments', 'DistilBERT', 'figures/', select_on_attribute=None)
    plot_all_mcb_algs('AmazonPolarity', 'ResNet-56', 'figures/', select_on_attribute=None)

    save_dir = 'figures/tables/'

    # Generate table for CelebA
    print('Generating tables for image datasets (Appendix G.2)')
    print(' '*4 + 'Generating table for CelebA: [ViT, ResNet-50]')
    generate_table_tabular_datasets(['ViT', 'ResNet-50'], 'CelebA', save_dir)
    # Generate table for Camelyon17
    print(' '*4 + 'Generating table for Camelyon17: [ViT, DenseNet-121]')
    generate_table_tabular_datasets(['ViT', 'DenseNet-121'], 'Camelyon17', save_dir)

    # Generate table for civil comments and amazon polarity
    print('Generating tables for language datasets (Appendix G.2)')
    print(' '*4 + 'Generating table for CivilComments: [DistilBERT]')
    generate_table_tabular_datasets(['DistilBERT'], 'CivilComments', save_dir)
    print(' '*4 + 'Generating table for AmazonPolarity: [ResNet-56]')
    generate_table_tabular_datasets(['ResNet-56'], 'AmazonPolarity', save_dir)


def data_reuse():
    '''
    Generate tables for tabular datasets with data reuse.
    '''
    groups_collection = 'default'
    datasets = ['ACSIncome', 'BankMarketing', 'CreditDefault', 'HMDA', 'MEPS']
    models = ['MLP', 'RandomForest', 'SVM', 'LogisticRegression', 'DecisionTree', 'NaiveBayes']

    print(f'Generating tables and figures for tabular datasets with {groups_collection} groups.')

    for dataset in datasets:
        # generate tables
        save_dir = 'figures/tables/data_reuse/'
        data_reuse_table_tabular_datasets(models, dataset, save_dir, results_dir='results/data_reuse/')


if __name__ == '__main__':
    tabular('default')
    tabular('alternate')
    non_tabular()
    data_reuse()
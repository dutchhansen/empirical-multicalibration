import os
import sys
# move to parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.results import experiments_table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 15)
plt.rcParams['lines.markersize'] = 8

# colorblindness-friendly palette
# see https://davidmathlogic.com/colorblind/#%233752B8-%234DAF4A-%23CC0004-%23904E9A-%23FF9500
# best not to use orange and blue as contrasting colors with this palette
CB_BLUE = '#3752B8'
CB_GREEN = '#4DAf4A'
CB_RED = '#CC0004'
CB_PURPLE = '#904E9A'
CB_ORANGE = '#FF9500'
CB_COLORS = [CB_BLUE, CB_GREEN, CB_PURPLE, CB_ORANGE]
SIMPLE_MODELS = ['RandomForest', 'LogisticRegression', 'MLP', 'DecisionTree', 'NaiveBayes', 'SVM']
DATASETS = ['CreditDefault', 'MEPS', 'BankMarketing', 'HMDA', 'ACSIncome']
DATASET_SIZES = {'ACSIncome': 200000, 
                 'BankMarketing': 45000, 
                 'CreditDefault': 30000, 
                 'HMDA': 114000, 
                 'MEPS': 11000, 
                 'CelebA': 200000, 
                 'Camelyon17': 450000, 
                 'CivilComments': 450000, 
                 'AmazonPolarity': 400000}


def format_ticks(ax, decimals=2):
    '''
    Rounds the x and y ticks to 2 decimal places,
    adjusts the font size of the ticks and labels,
    and sets the limits of the plot to the original values.
    '''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # matplotlib requires fixed ticks before setting labels, else warning thrown
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{tick:.{decimals}f}' for tick in ax.get_xticks()], fontsize=12)
    ax.set_yticklabels([f'{tick:.{decimals}f}' for tick in ax.get_yticks()], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def keep_CF_1_rows(df):
    df_restricted = df[df['calib_frac'] == 1.0]
    df_restricted = df_restricted[df_restricted['alg_type'].isin(['HKRR', 'HJZ'])]
    return df_restricted


def increasing_CF_helps_mcb_algs(dataset, model, SAVE_DIR, select_on_attribute='smECE/max', results_dir='results/'):
    df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
    df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)

    # Check if calib_frac includes 1.0
    if model in SIMPLE_MODELS:
        if 1.0 not in df_val['calib_frac'].unique():
            # Fetch the results for CF = 1.0 from MLP
            df_val_MLP, df_std_val_MLP = experiments_table('MLP', dataset, split='validation', results_dir=results_dir)

            # add all rows with CF = 1.0 to df_val
            df_val = pd.concat([df_val, keep_CF_1_rows(df_val_MLP)])
            df_std_val = pd.concat([df_std_val, keep_CF_1_rows(df_std_val_MLP)])

            # Do the same with test
            df_test_MLP, df_std_test_MLP = experiments_table('MLP', dataset, split='test', results_dir=results_dir)
            df_test = pd.concat([df_test, keep_CF_1_rows(df_test_MLP)])
            df_std_test = pd.concat([df_std_test, keep_CF_1_rows(df_std_test_MLP)])

    # sns.set_style("darkgrid", {"axes.facecolor": "0.94"})

    # Set the color palette to something nice
    sns.set_palette("colorblind")

    # Plot the pareto frontier
    fig, ax = plt.subplots()
    ax.set_facecolor('0.94')

    # Make the plot pretty with seaborn
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(False)
    # ax.set_facecolor('white')
    # Change grid line color to dark gray
    ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # print("std df")
    # print(df_std_test)

    if select_on_attribute == 'smECE/max':
        higher_is_better = False
    elif select_on_attribute == 'acc/agg':
        higher_is_better = True
    else:
        raise ValueError(f'select_on_attribute must be one of ["smECE/max", "acc/agg"]')

    for cf in df_val['calib_frac'].unique():
        # No mcb algorithms for cf = 0.0
        if cf == 0.0 or cf == 0.01:
            continue

        best_param_id_val = best_param_id(df_val, cf, 'HKRR', select_on_attribute, higher_is_better=higher_is_better)
        selected_row_test = get_row_with_param_id(df_test, cf, 'HKRR', best_param_id_val)
        selected_row_test_std = get_row_with_param_id(df_std_test, cf, 'HKRR', best_param_id_val)
        # Plot the HKRR point with std error bars on the test set
        ax.errorbar(cf, 
                    selected_row_test['smECE/max'], 
                    yerr=selected_row_test_std['smECE/max'], 
                    fmt='o',
                    label='HKRR' if cf == 1.0 else "",
                    color=CB_BLUE,
                    capsize=4, capthick=2, elinewidth=2, markersize=8)
        
        best_param_id_val = best_param_id(df_val, cf, 'HJZ', select_on_attribute, higher_is_better=False)
        selected_row_test = get_row_with_param_id(df_test, cf, 'HJZ', best_param_id_val)
        selected_row_test_std = get_row_with_param_id(df_std_test, cf, 'HJZ', best_param_id_val)
        # Plot the HJZ point with std error bars on the test set
        ax.errorbar(cf,
                    selected_row_test['smECE/max'],
                    yerr=selected_row_test_std['smECE/max'],
                    fmt='^',
                    label='HJZ' if cf == 1.0 else "",
                    color=CB_GREEN,
                    capsize=4, capthick=2, elinewidth=2, markersize=8)

    erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
    erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)

    # Draw a horizontal line at the ERM smECE/max
    x = np.linspace(0, 1, 100)
    y = np.full_like(x, erm_row['smECE/max'])
    err = np.full_like(x, erm_row_std['smECE/max'])

    plt.plot(x, y, color=CB_RED, linestyle='--', label='ERM')
    plt.fill_between(x, y-err, y+err, color=CB_RED, alpha=0.1, lw=0)

    # Add a legend
    ax.legend(loc='upper right')

    # Make the x-axis labels larger
    # Round the x and y ticks to 2 decimal places
    format_ticks(ax, decimals=2)

    # Make the x and y labels larger
    ax.set_xlabel('Calibration Fraction', fontsize=20)
    ax.set_ylabel('Max Group smECE', fontsize=20)
    # ax.set_title(f'Accuracy vs Max Group smECE for MLP trained on {dataset}')  

    # Color the legend background dark gray
    legend = ax.legend()
    legend.get_frame().set_facecolor('0.8')
    legend.get_frame().set_linewidth(0.0)

    # Make the legend font size 16
    for label in legend.get_texts():
        label.set_fontsize('16')

    # if dataset == 'CreditDefault':
        # ax.set_xlim(left=0.8005)

    plt.title('{0} on {1}'.format(model, dataset), fontsize=20)

    plt.tight_layout()

    if SAVE_DIR is not None:
        # Check if SAVE_DIR/model exists, if not create it
        if not os.path.exists(SAVE_DIR + f'{model}/'):
            os.makedirs(SAVE_DIR + f'{model}/')

        plt.savefig(SAVE_DIR + f'{model}/calfrac_{dataset}_selecton_{select_on_attribute[:3]}.pdf')

    else:
        plt.show()

    # restore memory
    plt.close()


def increasing_CF_hurts_acc(dataset, model, SAVE_DIR, select_on_attribute='smECE/max', results_dir='results/'):
    df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
    df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)

    # Check if calib_frac includes 1.0
    if model in SIMPLE_MODELS:
        if 1.0 not in df_val['calib_frac'].unique():
            # Fetch the results for CF = 1.0 from MLP
            df_val_MLP, df_std_val_MLP = experiments_table('MLP', dataset, split='validation', results_dir=results_dir)

            # add all rows with CF = 1.0 to df_val
            df_val = pd.concat([df_val, keep_CF_1_rows(df_val_MLP)])
            df_std_val = pd.concat([df_std_val, keep_CF_1_rows(df_std_val_MLP)])

            # Do the same with test
            df_test_MLP, df_std_test_MLP = experiments_table('MLP', dataset, split='test', results_dir=results_dir)
            df_test = pd.concat([df_test, keep_CF_1_rows(df_test_MLP)])
            df_std_test = pd.concat([df_std_test, keep_CF_1_rows(df_std_test_MLP)])

    # Set the color palette to something nice
    sns.set_palette("colorblind")

    # Plot the pareto frontier
    fig, ax = plt.subplots()
    ax.set_facecolor('0.94')

    # Make the plot pretty with seaborn
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(False)
    # ax.set_facecolor('white')
    # Change grid line color to dark gray
    ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    # print("std df")
    # print(df_std_test)

    if select_on_attribute == 'smECE/max':
        higher_is_better = False
    elif select_on_attribute == 'acc/agg':
        higher_is_better = True
    else:
        raise ValueError(f'select_on_attribute must be one of ["smECE/max", "acc/agg"]')

    for cf in df_val['calib_frac'].unique():
        # No mcb algorithms for cf = 0.0
        if cf == 0.0 or cf == 0.01:
            continue

        best_param_id_val = best_param_id(df_val, cf, 'HKRR', select_on_attribute, higher_is_better=higher_is_better)
        selected_row_test = get_row_with_param_id(df_test, cf, 'HKRR', best_param_id_val)
        selected_row_test_std = get_row_with_param_id(df_std_test, cf, 'HKRR', best_param_id_val)
        # Plot the HKRR point with std error bars on the test set
        ax.errorbar(cf, 
                    selected_row_test['acc/agg'], 
                    yerr=selected_row_test_std['acc/agg'], 
                    fmt='o',
                    label='HKRR' if cf == 1.0 else "",
                    color=CB_BLUE,
                    capsize=4, capthick=2, elinewidth=2, markersize=8)
        
        best_param_id_val = best_param_id(df_val, cf, 'HJZ', select_on_attribute, higher_is_better=False)
        selected_row_test = get_row_with_param_id(df_test, cf, 'HJZ', best_param_id_val)
        selected_row_test_std = get_row_with_param_id(df_std_test, cf, 'HJZ', best_param_id_val)
        # Plot the HJZ point with std error bars on the test set
        ax.errorbar(cf, 
                    selected_row_test['acc/agg'],
                    yerr=selected_row_test_std['acc/agg'],
                    fmt='^',
                    label='HJZ' if cf == 1.0 else "",
                    color=CB_GREEN,
                    capsize=4, capthick=2, elinewidth=2, markersize=8)

    erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
    erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)

    # Draw a horizontal line at the ERM smECE/max
    x = np.linspace(0, 1, 100)
    y = np.full_like(x, erm_row['acc/agg'])
    err = np.full_like(x, erm_row_std['acc/agg'])

    plt.plot(x, y, color=CB_RED, linestyle='--', label='ERM')
    plt.fill_between(x, y-err, y+err, color=CB_RED, alpha=0.1, lw=0)

    # Add a legend
    ax.legend(loc='upper right')

    # Make the x-axis labels larger
    # Round the x and y ticks to 2 decimal places
    format_ticks(ax, decimals=2)

    # Make the x and y labels larger
    ax.set_xlabel('Calibration Fraction', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    # ax.set_title(f'Accuracy vs Max Group smECE for MLP trained on {dataset}')  

    # Color the legend background dark gray
    legend = ax.legend()
    legend.get_frame().set_facecolor('0.8')
    legend.get_frame().set_linewidth(0.0)

    # Make the legend font size 16
    for label in legend.get_texts():
        label.set_fontsize('16')

    plt.title('{0} on {1}'.format(model, dataset), fontsize=20)

    plt.tight_layout()

    if SAVE_DIR is not None:
        # Check if SAVE_DIR/model exists, if not create it
        if not os.path.exists(SAVE_DIR + f'{model}/'):
            os.makedirs(SAVE_DIR + f'{model}/')

        plt.savefig(SAVE_DIR + f'{model}/calfrac_accuracy_{dataset}_selecton_{select_on_attribute[:3]}.pdf')

    else:
        plt.show()

    # restore memory
    plt.close()


def best_param_id(df, cf, alg_type, metric, higher_is_better=True):
    df_cf = df[df['calib_frac'] == cf]
    df_cf = df_cf[df_cf['alg_type'] == alg_type]

    # Sort  the dataframe by column defined by metric in descending order
    ascending = True
    if higher_is_better:
        ascending = False
    
    df_cf = df_cf.sort_values(by=metric, ascending=ascending)
    return df_cf['param_id'].values[0]


def get_row_with_param_id(df, cf, alg_type, param_id):
    return df[(df['calib_frac'] == cf) & (df['alg_type'] == alg_type) & (df['param_id'] == param_id)]


def get_best_param_id_cf(df, alg_type, metric, higher_is_better=True):
    df = df[df['alg_type'] == alg_type]

    # Sort  the dataframe by column defined by metric in descending order
    ascending = True
    if higher_is_better:
        ascending = False
    
    df = df.sort_values(by=metric, ascending=ascending)

    # return the param_id of the best performing parameter
    cf = df['calib_frac'].values[0]
    id = df['param_id'].values[0]

    return cf, id


def plot_all_mcb_algs(dataset, model, SAVE_DIR, select_on_attribute=None, results_dir='results/'):
    df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
    df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)
    
    # Check if calib_frac includes 1.0
    if model in SIMPLE_MODELS:
        if 1.0 not in df_val['calib_frac'].unique():
            # Fetch the results for CF = 1.0 from MLP
            df_val_MLP, df_std_val_MLP = experiments_table('MLP', dataset, split='validation', results_dir=results_dir)

            # add all rows with CF = 1.0 to df_val
            df_val = pd.concat([df_val, keep_CF_1_rows(df_val_MLP)])
            df_std_val = pd.concat([df_std_val, keep_CF_1_rows(df_std_val_MLP)])

            # Do the same with test
            df_test_MLP, df_std_test_MLP = experiments_table('MLP', dataset, split='test', results_dir=results_dir)
            df_test = pd.concat([df_test, keep_CF_1_rows(df_test_MLP)])
            df_std_test = pd.concat([df_std_test, keep_CF_1_rows(df_std_test_MLP)])


    fig, ax = plt.subplots()

    ax.set_facecolor('0.94')
    # Set the color palette to something nice
    # sns.set_palette("colorblind")

    # Make the plot pretty with seaborn
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Change grid line color to dark gray
    ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    for calib_method in ['HKRR', 'HJZ']:
        label_point = True
        for cf in df_val['calib_frac'].unique():
            if cf == 0.0 or cf == 0.01:
                continue
            if select_on_attribute:
                best_param_id_val = best_param_id(df_val, cf, calib_method, select_on_attribute, higher_is_better=False)
                selected_row_test = get_row_with_param_id(df_test, cf, calib_method, best_param_id_val)
                selected_row_test_std = get_row_with_param_id(df_std_test, cf, calib_method, best_param_id_val)
                ax.errorbar(selected_row_test['acc/agg'],
                            selected_row_test['smECE/max'],
                            xerr=selected_row_test_std['acc/agg'],
                            yerr=selected_row_test_std['smECE/max'],
                            fmt='o' if calib_method == 'HKRR' else '^',
                            label=calib_method if label_point else "",
                            color=CB_BLUE if calib_method == 'HKRR' else CB_GREEN,
                            capsize=4, capthick=2, elinewidth=2, markersize=8)
                label_point = False

            else:
                # Plot every calibration method
                # Get all unique_param_ids for this calib_method
                unique_param_ids = df_val[(df_val['calib_frac'] == cf) & (df_val['alg_type'] == calib_method)]['param_id'].unique()
                for param_id in unique_param_ids:
                    selected_row_test = get_row_with_param_id(df_test, cf, calib_method, param_id)
                    # selected_row_test_std = get_row_with_param_id(df_std_test, cf, calib_method, param_id)
                    ax.plot(selected_row_test['acc/agg'],
                            selected_row_test['smECE/max'],
                            label=calib_method if label_point else "",
                            color=CB_BLUE if calib_method == 'HKRR' else CB_GREEN,
                            alpha=0.8,
                            marker='o' if calib_method == 'HKRR' else '^', 
                            markersize=8)
            
                    label_point = False

    # Plot ERM
    erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
    erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)
    ax.errorbar(erm_row['acc/agg'], 
                erm_row['smECE/max'], 
                xerr=erm_row_std['acc/agg'],
                yerr=erm_row_std['smECE/max'],
                fmt='o',
                label='ERM', 
                color=CB_RED,
                capsize=2, capthick=2, elinewidth=2, markersize=8)


    ax.legend()

    # Make the x-axis labels larger
    # Round the ticks
    if model == 'ViT':
        format_ticks(ax, decimals=3)
    else:
        format_ticks(ax, decimals=2)

    # Make the x and y labels larger
    ax.set_xlabel('Accuracy', fontsize=20)
    ax.set_ylabel('Max Group smECE', fontsize=20)
    # ax.set_title(f'Accuracy vs Max Group smECE for MLP trained on {dataset}')

    # Color the legend background dark gray
    legend = ax.legend()
    legend.get_frame().set_facecolor('0.8')
    legend.get_frame().set_linewidth(0.0)

    # Make the legend font size 16
    for label in legend.get_texts():
        label.set_fontsize('16')
    
    plt.title('{0} on {1}'.format(model, dataset), fontsize=20)

    plt.tight_layout()

    if SAVE_DIR is not None:
        # Check if SAVE_DIR/model exists, if not create it
        if not os.path.exists(SAVE_DIR + f'{model}/'):
            os.makedirs(SAVE_DIR + f'{model}/')

        if select_on_attribute:
            plt.savefig(SAVE_DIR + f'{model}/all_mcb_algs_{dataset}_selecton_{select_on_attribute[:3]}.pdf')
        else:
            plt.savefig(SAVE_DIR + f'{model}/all_mcb_algs_{dataset}.pdf')

    else:
        plt.show()

    # restore memory
    plt.close()


def print_best_params(dataset, model, alg_type, select_on_attribute='smECE/max', results_dir='results/'):
    df_val, _ = experiments_table(model, dataset, split='validation', results_dir=results_dir)

    # df_val = df_val[df_val['calib_frac'] != 0.01]
    df_val = df_val[df_val['calib_frac'] != 0.0]
    df_val = df_val[df_val['alg_type'] == alg_type]

    if select_on_attribute == 'smECE/max':
        # Print the best smECE/max in the df_val dataframe, as well as the corresponding cf and param_id
        df_val = df_val.sort_values(by='smECE/max', ascending=True)
        # Now print ONLY the cf and param_id of the best performing parameter
        print(df_val[['alg_type','smECE/max', 'calib_frac', 'param_id']].head(1))
    elif select_on_attribute == 'acc/agg':
        df_val = df_val.sort_values(by='acc/agg', ascending=False)
        print(df_val[['alg_type', 'acc/agg', 'calib_frac', 'param_id']].head(1))
    else:
        raise ValueError(f'select_on_attribute must be one of ["smECE/max", "acc/agg"]')


def print_HKRR_dependence_on_alpha(dataset, model, results_dir='results/'):
    df_val, df_std_val, param_dict = experiments_table(model, dataset, split='validation', return_param_dict=True, results_dir=results_dir)
    erm_row = get_row_with_param_id(df_val, 0.0, 'ERM', 0)
    erm_row_std = get_row_with_param_id(df_std_val, 0.0, 'ERM', 0)

    df_val = df_val[df_val['alg_type'] == 'HKRR']
    # Drop the 0.01 calibration fraction
    df_val = df_val[df_val['calib_frac'] != 0.01]

    df_val = df_val.sort_values(by='smECE/max', ascending=True)

    print('validation')
    print(df_val[['alg_type', 'smECE/max', 'calib_frac', 'param_id']])

    # Plot smECE/max vs lambda
    fig, ax = plt.subplots()

    # For each row except the ERM row, plot the smECE/max vs lambda
    for idx, row in df_val.iterrows():
        if row['calib_frac'] == 0.0:
            continue
        ax.errorbar(param_dict[row['calib_frac']]['HKRR'][row['param_id']]['alpha'],
                    row['smECE/max'],
                    fmt='o',
                    label=f'CF={row["calib_frac"]}')

    # Draw a horizontal line at the ERM smECE/max
    # get min and max x of plotted points from matplotlib
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, 100)
    y = np.full_like(x, erm_row['smECE/max'])
    err = np.full_like(x, erm_row_std['smECE/max'])

    plt.plot(x, y, color=CB_RED, linestyle='--', label='ERM')
    plt.fill_between(x, y-err, y+err, color=CB_RED, alpha=0.1, lw=0)

    plt.xlabel('alpha')
    plt.ylabel('Max Group smECE')
    plt.title(f'{model} on {dataset} HKRR smECE/max vs Lambda')
    # plt.legend()
    plt.show()

    # restore memory
    plt.close()


def plot_calib_vs_group_calib_error(models, dataset, SAVE_DIR, results_dir='results/'):
    
    fig, ax = plt.subplots()

    ax.set_facecolor('0.94')
    # Set the color palette to something nice
    # sns.set_palette("colorblind")

    # Make the plot pretty with seaborn
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ax.spines['left'].set_linewidth(0.5)
    # ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Change grid line color to dark gray
    ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)

    first_erm = True
    label_point_HKRR = True
    label_point_HJZ = True
    for model in models:
        df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
        df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)

        for calib_method in ['HKRR', 'HJZ']:
            for cf in df_val['calib_frac'].unique():
                if cf == 0.0 or cf == 0.01:
                    continue
                # Plot every calibration method
                # Get all unique_param_ids for this calib_method
                unique_param_ids = df_val[(df_val['calib_frac'] == cf) & (df_val['alg_type'] == calib_method)]['param_id'].unique()
                for param_id in unique_param_ids:
                    selected_row_test = get_row_with_param_id(df_test, cf, calib_method, param_id)
                    # selected_row_test_std = get_row_with_param_id(df_std_test, cf, calib_method, param_id)
                    if calib_method == 'HKRR' and label_point_HKRR:
                        label = 'HKRR'
                        label_point_HKRR = False
                    elif calib_method == 'HJZ' and label_point_HJZ:
                        label = 'HJZ'
                        label_point_HJZ = False
                    else:
                        label = ""
                    ax.plot(selected_row_test['smECE/agg'],
                            selected_row_test['smECE/max'],
                            label=label,
                            color=CB_BLUE if calib_method == 'HKRR' else CB_GREEN,
                            alpha=0.8,
                            marker='o' if calib_method == 'HKRR' else '^', 
                            markersize=8)

            # Plot ERM
            erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
            erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)
            ax.plot(erm_row['smECE/agg'], 
                        erm_row['smECE/max'], 
                        label='ERM' if first_erm else "", 
                        color=CB_RED,
                        marker='o', 
                        markersize=8)
            first_erm = False


    ax.legend()

    # Make the x-axis labels larger
    # Round the ticks
    if model == 'ViT':
        format_ticks(ax, decimals=3)
    else:
        format_ticks(ax, decimals=2)

    # Make the x and y labels larger
    ax.set_xlabel('Overall smECE', fontsize=20)
    ax.set_ylabel('Max Group smECE', fontsize=20)
    # ax.set_title(f'Accuracy vs Max Group smECE for MLP trained on {dataset}')

    # Color the legend background dark gray
    legend = ax.legend()
    legend.get_frame().set_facecolor('0.8')
    legend.get_frame().set_linewidth(0.0)

    # Make the legend font size 16
    for label in legend.get_texts():
        label.set_fontsize('16')
    
    plt.title('Calibration vs Multicalibration Errors on {0}'.format(dataset), fontsize=20)

    plt.tight_layout()

    if SAVE_DIR is not None:
        # Check if SAVE_DIR/model exists, if not create it
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        plt.savefig(SAVE_DIR + 'all_mcb_algs_{0}.pdf'.format(dataset))

    else:
        plt.show()

    # restore memory
    plt.close()


def dataset_size_vs_mcb_error(models_and_datasets, SAVE_DIR, results_dir='results/'):
    fig, ax = plt.subplots()

    # sns.set_style("darkgrid", {"axes.facecolor": "0.94"})

    ax.set_facecolor('0.94')
    # Set the color palette to something nice
    # sns.set_palette("colorblind")

    # Make the plot pretty with seaborn
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ax.spines['left'].set_linewidth(0.5)
    # ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Change grid line color to dark gray
    ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)
    first_erm = True
    label_point_HKRR = True
    label_point_HJZ = True

    for model, datasets in models_and_datasets:
        print(model, datasets)
        for dataset in datasets:
            df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
            df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)


            for calib_method in ['HKRR', 'HJZ']:
                for cf in df_val['calib_frac'].unique():
                    if cf == 0.0 or cf == 0.01:
                        continue
                cf, best_param_id = get_best_param_id_cf(df_val, calib_method, 'smECE/max', higher_is_better=False)
                selected_row_test = get_row_with_param_id(df_test, cf, calib_method, best_param_id)
                selected_row_test_std = get_row_with_param_id(df_std_test, cf, calib_method, best_param_id)
                if calib_method == 'HKRR' and label_point_HKRR:
                    label = 'HKRR'
                    label_point_HKRR = False
                elif calib_method == 'HJZ' and label_point_HJZ:
                    label = 'HJZ'
                    label_point_HJZ = False
                else:
                    label = ""

                ax.errorbar(DATASET_SIZES[dataset],
                            selected_row_test['smECE/max'],
                            yerr=selected_row_test_std['smECE/max'],
                            fmt='o' if calib_method == 'HKRR' else '^',
                            label=label,
                            color=CB_BLUE if calib_method == 'HKRR' else CB_GREEN)            

            # Get erm
            erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
            erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)
            ax.errorbar(DATASET_SIZES[dataset],
                        erm_row['smECE/max'],
                        yerr=erm_row_std['smECE/max'],
                        fmt='o',
                        label='ERM' if first_erm else "",
                        color=CB_RED)
            first_erm = False


    # Make the x-axis labels larger
    # Round the ticks
    if model == 'ViT':
        format_ticks(ax, decimals=3)
    else:
        format_ticks(ax, decimals=2)

    # Make the x and y labels larger
    ax.set_xlabel('Dataset Size', fontsize=20)
    ax.set_ylabel('Max Group smECE', fontsize=20)
    # ax.set_title(f'Accuracy vs Max Group smECE for MLP trained on {dataset}')

    # Color the legend background dark gray
    legend = ax.legend()
    legend.get_frame().set_facecolor('0.8')
    legend.get_frame().set_linewidth(0.0)

    # Make the legend font size 16
    for label in legend.get_texts():
        label.set_fontsize('16')
    
    plt.title('Calibration vs Multicalibration Errors on {0}'.format(dataset), fontsize=20)

    plt.tight_layout()

    if SAVE_DIR is not None:
        # Check if SAVE_DIR/model exists, if not create it
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        plt.savefig(SAVE_DIR + 'data_size_vs_mcb.pdf'.format(dataset))

    else:
        plt.show()

    # restore memory
    plt.close()

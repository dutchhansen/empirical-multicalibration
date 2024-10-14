import os
import sys
# move to parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.plots import get_row_with_param_id, get_best_param_id_cf, keep_CF_1_rows
from scripts.results import experiments_table
import pandas as pd

# import mcb algorithms config object
from scripts.results import DEFAULT_MCB_ALGS
TAB_DATA = ['HMDA', 'ACSIncome', 'BankMarketing', 'CreditDefault', 'MEPS']


def str_format(val, std, n_decimals=3, bold=False):
    # First round val and std to n_decimals
    val = round(val, n_decimals)
    std = round(std, n_decimals)
    if bold:
        final_str = "\\highest{{{0} ± {1}}}".format(val, std)
    else:
        final_str = "{0} ± {1}".format(val, std)
    
    # print('input', val, std, bold)
    # print('final string here', final_str)
    return final_str


def generate_table_tabular_datasets(models, dataset, SAVE_DIR, results_dir='results/'):
    # create dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    header = ['\\textbf{{Model}}', 
              '\\textbf{{ECE}} $\\downarrow$', 
              '\\textbf{{Max ECE}} $\\downarrow$', 
              '\\textbf{{smECE}} $\\downarrow$', 
              '\\textbf{{Max smECE}} $\\downarrow$', 
              '\\textbf{{Acc}} $\\uparrow$']

    # Initialize pandas array with header
    result_df = pd.DataFrame(columns=header)

    for model in models:
        df_val, df_std_val = experiments_table(model, dataset, split='validation', results_dir=results_dir)
        df_test, df_std_test = experiments_table(model, dataset, split='test', results_dir=results_dir)

        # Add CF 1.0 row if necessary
        if 1.0 not in df_val['calib_frac'].unique() and dataset in TAB_DATA:
            # Fetch the results for CF = 1.0 from MLP
            df_val_MLP, df_std_val_MLP = experiments_table('MLP', dataset, split='validation', results_dir=results_dir)

            # add all rows with CF = 1.0 to df_val
            df_val = pd.concat([df_val, keep_CF_1_rows(df_val_MLP)])
            df_std_val = pd.concat([df_std_val, keep_CF_1_rows(df_std_val_MLP)])

            # Do the same with test
            df_test_MLP, df_std_test_MLP = experiments_table('MLP', dataset, split='test', results_dir=results_dir)
            df_test = pd.concat([df_test, keep_CF_1_rows(df_test_MLP)])
            df_std_test = pd.concat([df_std_test, keep_CF_1_rows(df_std_test_MLP)])


        # Get the performance of the ERM model
        erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
        erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)

        # Store the results in an array
        erm_arr = [model + ' ERM', 
                   erm_row['ECE/agg'].values[0], 
                   erm_row['ECE/max'].values[0], 
                   erm_row['smECE/agg'].values[0], 
                   erm_row['smECE/max'].values[0], 
                   erm_row['acc/agg'].values[0]]

        erm_arr_std = [model + ' ERM', 
                       erm_row_std['ECE/agg'].values[0], 
                       erm_row_std['ECE/max'].values[0], 
                       erm_row_std['smECE/agg'].values[0], 
                       erm_row_std['smECE/max'].values[0], 
                       erm_row_std['acc/agg'].values[0]]

        # Get the best performing HKRR row
        cf, id = get_best_param_id_cf(df_val, 'HKRR', 'smECE/max', higher_is_better=False)
        best_row_test_HKRR = get_row_with_param_id(df_test, cf, 'HKRR', id)
        best_row_test_HKRR_std = get_row_with_param_id(df_std_test, cf, 'HKRR', id)
        hkrr_arr = [model + ' $\\hkrr$', 
                    best_row_test_HKRR['ECE/agg'].values[0],
                    best_row_test_HKRR['ECE/max'].values[0],
                    best_row_test_HKRR['smECE/agg'].values[0],
                    best_row_test_HKRR['smECE/max'].values[0],
                    best_row_test_HKRR['acc/agg'].values[0]]
        
        hkrr_arr_std = [model + ' HKRR',
                        best_row_test_HKRR_std['ECE/agg'].values[0],
                        best_row_test_HKRR_std['ECE/max'].values[0],
                        best_row_test_HKRR_std['smECE/agg'].values[0],
                        best_row_test_HKRR_std['smECE/max'].values[0],
                        best_row_test_HKRR_std['acc/agg'].values[0]]

        # Get the best performing HJZ row
        cf, id = get_best_param_id_cf(df_val, 'HJZ', 'smECE/max', higher_is_better=False)
        best_row_test_HJZ = get_row_with_param_id(df_test, cf, 'HJZ', id)
        best_row_test_HJZ_std = get_row_with_param_id(df_std_test, cf, 'HJZ', id)
        hjz_arr = [model + ' $\\hjz$',
                    best_row_test_HJZ['ECE/agg'].values[0],
                    best_row_test_HJZ['ECE/max'].values[0],
                    best_row_test_HJZ['smECE/agg'].values[0],
                    best_row_test_HJZ['smECE/max'].values[0],
                    best_row_test_HJZ['acc/agg'].values[0]]

        hjz_arr_std = [model + ' HJZ',
                          best_row_test_HJZ_std['ECE/agg'].values[0],
                          best_row_test_HJZ_std['ECE/max'].values[0],
                          best_row_test_HJZ_std['smECE/agg'].values[0],
                          best_row_test_HJZ_std['smECE/max'].values[0],
                          best_row_test_HJZ_std['acc/agg'].values[0]]
        
        # Get the best performing Platt row
        cf, id = get_best_param_id_cf(df_val, 'Platt', 'smECE/max', higher_is_better=False)
        best_row_test_platt = get_row_with_param_id(df_test, cf, 'HJZ', id)
        best_row_test_platt_std = get_row_with_param_id(df_std_test, cf, 'HJZ', id)
        platt_arr = [model + ' Platt',
                    best_row_test_platt['ECE/agg'].values[0],
                    best_row_test_platt['ECE/max'].values[0],
                    best_row_test_platt['smECE/agg'].values[0],
                    best_row_test_platt['smECE/max'].values[0],
                    best_row_test_platt['acc/agg'].values[0]]
        
        platt_arr_std = [model + ' Platt',
                        best_row_test_platt_std['ECE/agg'].values[0],
                        best_row_test_platt_std['ECE/max'].values[0],
                        best_row_test_platt_std['smECE/agg'].values[0],
                        best_row_test_platt_std['smECE/max'].values[0],
                        best_row_test_platt_std['acc/agg'].values[0]]

        # Get the best performing Temp row
        cf, id = get_best_param_id_cf(df_val, 'Temp', 'smECE/max', higher_is_better=False)
        best_row_test_temp = get_row_with_param_id(df_test, cf, 'Temp', id)
        best_row_test_temp_std = get_row_with_param_id(df_std_test, cf, 'Temp', id)
        temp_arr = [model + ' Temp',
                    best_row_test_temp['ECE/agg'].values[0],
                    best_row_test_temp['ECE/max'].values[0],
                    best_row_test_temp['smECE/agg'].values[0],
                    best_row_test_temp['smECE/max'].values[0],
                    best_row_test_temp['acc/agg'].values[0]]
        
        temp_arr_std = [model + ' Temp',
                        best_row_test_temp_std['ECE/agg'].values[0],
                        best_row_test_temp_std['ECE/max'].values[0],
                        best_row_test_temp_std['smECE/agg'].values[0],
                        best_row_test_temp_std['smECE/max'].values[0],
                        best_row_test_temp_std['acc/agg'].values[0]]
        

        # Get the best performing Isotonic row
        cf, id = get_best_param_id_cf(df_val, 'Isotonic', 'smECE/max', higher_is_better=False)
        best_row_test_isotonic = get_row_with_param_id(df_test, cf, 'Isotonic', id)
        best_row_test_isotonic_std = get_row_with_param_id(df_std_test, cf, 'Isotonic', id)
        isotonic_arr = [model + ' Isotonic',
                        best_row_test_isotonic['ECE/agg'].values[0],
                        best_row_test_isotonic['ECE/max'].values[0],
                        best_row_test_isotonic['smECE/agg'].values[0],
                        best_row_test_isotonic['smECE/max'].values[0],
                        best_row_test_isotonic['acc/agg'].values[0]]
        
        isotonic_arr_std = [model + ' Isotonic',
                            best_row_test_isotonic_std['ECE/agg'].values[0],
                            best_row_test_isotonic_std['ECE/max'].values[0],
                            best_row_test_isotonic_std['smECE/agg'].values[0],
                            best_row_test_isotonic_std['smECE/max'].values[0],
                            best_row_test_isotonic_std['acc/agg'].values[0]]

        # print(erm_arr)
        # print(hkrr_arr)
        # print(hjz_arr)

        # For each column except the first, find the smallest value and store it in a boolean array
        arr_best = [None] + [min(erm_arr[i], hkrr_arr[i], hjz_arr[i], platt_arr[i], temp_arr[i], isotonic_arr[i]) for i in range(1, 5)] + [max(erm_arr[5], hkrr_arr[5], hjz_arr[5], platt_arr[5], temp_arr[5], isotonic_arr[5])]

        # Convert to the appropriate strings
        erm_final = [model + ' ERM'] + [str_format(erm_arr[i], erm_arr_std[i], bold=(erm_arr[i] == arr_best[i])) for i in range(1, 6)]
        hkrr_final = [model + ' $\\hkrr$'] + [str_format(hkrr_arr[i], hkrr_arr_std[i], bold=(hkrr_arr[i] == arr_best[i])) for i in range(1, 6)]
        hjz_final = [model + ' $\\hjz$'] + [str_format(hjz_arr[i], hjz_arr_std[i], bold=(hjz_arr[i] == arr_best[i])) for i in range(1, 6)]
        platt_final = [model + ' Platt'] + [str_format(platt_arr[i], platt_arr_std[i], bold=(platt_arr[i] == arr_best[i])) for i in range(1, 6)]
        temp_final = [model + ' Temp'] + [str_format(temp_arr[i], temp_arr_std[i], bold=(temp_arr[i] == arr_best[i])) for i in range(1, 6)]
        isotonic_final = [model + ' Isotonic'] + [str_format(isotonic_arr[i], isotonic_arr_std[i], bold=(isotonic_arr[i] == arr_best[i])) for i in range(1, 6)]

        # Append the arrays to the result_df
        result_df = result_df._append(pd.Series(erm_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(hkrr_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(hjz_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(platt_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(temp_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(isotonic_final, index=header), ignore_index=True)

    # Print the dataframe
    # print('DATAFRAME ======================')
    # print(result_df)

    # Save the datafram with to_latex
    result_df.to_latex(SAVE_DIR + f'{dataset}_table.tex', index=False, index_names=False)

    # Load the table as a string
    with open(SAVE_DIR + f'{dataset}_table.tex', 'r') as f:
        table_str = f.read()
    
    # After the 7th line, add a new line with only '\midrule' 
    table_str = table_str.split('\n')
    k = 0
    odd = -1
    for i in range(4, len(table_str) - 1, 1):
        # Prepend '\rowcolor{Wheat1}' to the line on alternating sets of 3  rows
        if odd == -1 and k < 6 and len(models) > 1:
            table_str[i] = '\\rowcolor{Wheat1} ' + table_str[i]
        # Between every set of 3 rows, add a '\midrule' to separate out the performance of different
        # base models
        if i > 6 and k == 6:
            table_str.insert(i, '\midrule')
            k = 0
            # flip odd
            odd *= -1
        else:
            k += 1

    # Save the string back into the file
    with open(SAVE_DIR + f'{dataset}_table.tex', 'w') as f:
        f.write('\n'.join(table_str))


def data_reuse_table_tabular_datasets(models, dataset, SAVE_DIR, results_dir='results/'):
    '''
    Unlike previous function, this admits a single calibration fraction, namely
    that of calib_frac=0 and calib_train_overlap=1.0. Generates table associated 
    with experiments examining the effect of data recycling in multicalibration methods.

    Main difference here is how we build config of mcb algorithms, and where
    we save the resulting table.
    '''

    # create dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    header = ['\\textbf{{Model}}', 
              '\\textbf{{ECE}} $\\downarrow$', 
              '\\textbf{{Max ECE}} $\\downarrow$', 
              '\\textbf{{smECE}} $\\downarrow$', 
              '\\textbf{{Max smECE}} $\\downarrow$', 
              '\\textbf{{Acc}} $\\uparrow$']

    # Initialize pandas array with header
    result_df = pd.DataFrame(columns=header)

    for model in models:

        # create alg config object
        # since recycling data, we only need the 0.0 calibration fraction
        # and we need to examime every calibration algorithm
        mcb_algs = DEFAULT_MCB_ALGS
        mcb_algs[0] = {
            **mcb_algs[0],
            **mcb_algs[0.4]}

        df_val, df_std_val = experiments_table(model, dataset, split='validation', alg_config=mcb_algs, results_dir=results_dir)
        df_test, df_std_test = experiments_table(model, dataset, split='test', alg_config=mcb_algs, results_dir=results_dir)

        # Get the performance of the ERM model
        erm_row = get_row_with_param_id(df_test, 0.0, 'ERM', 0)
        erm_row_std = get_row_with_param_id(df_std_test, 0.0, 'ERM', 0)

        # Store the results in an array
        erm_arr = [model + ' ERM', 
                   erm_row['ECE/agg'].values[0], 
                   erm_row['ECE/max'].values[0], 
                   erm_row['smECE/agg'].values[0], 
                   erm_row['smECE/max'].values[0], 
                   erm_row['acc/agg'].values[0]]

        erm_arr_std = [model + ' ERM', 
                       erm_row_std['ECE/agg'].values[0], 
                       erm_row_std['ECE/max'].values[0], 
                       erm_row_std['smECE/agg'].values[0], 
                       erm_row_std['smECE/max'].values[0], 
                       erm_row_std['acc/agg'].values[0]]

        # Get the best performing HKRR row
        cf, id = get_best_param_id_cf(df_val, 'HKRR', 'smECE/max', higher_is_better=False)
        best_row_test_HKRR = get_row_with_param_id(df_test, cf, 'HKRR', id)
        best_row_test_HKRR_std = get_row_with_param_id(df_std_test, cf, 'HKRR', id)
        hkrr_arr = [model + ' $\\hkrr$', 
                    best_row_test_HKRR['ECE/agg'].values[0],
                    best_row_test_HKRR['ECE/max'].values[0],
                    best_row_test_HKRR['smECE/agg'].values[0],
                    best_row_test_HKRR['smECE/max'].values[0],
                    best_row_test_HKRR['acc/agg'].values[0]]
        
        hkrr_arr_std = [model + ' HKRR',
                        best_row_test_HKRR_std['ECE/agg'].values[0],
                        best_row_test_HKRR_std['ECE/max'].values[0],
                        best_row_test_HKRR_std['smECE/agg'].values[0],
                        best_row_test_HKRR_std['smECE/max'].values[0],
                        best_row_test_HKRR_std['acc/agg'].values[0]]

        # Get the best performing HJZ row
        cf, id = get_best_param_id_cf(df_val, 'HJZ', 'smECE/max', higher_is_better=False)
        best_row_test_HJZ = get_row_with_param_id(df_test, cf, 'HJZ', id)
        best_row_test_HJZ_std = get_row_with_param_id(df_std_test, cf, 'HJZ', id)
        hjz_arr = [model + ' $\\hjz$',
                    best_row_test_HJZ['ECE/agg'].values[0],
                    best_row_test_HJZ['ECE/max'].values[0],
                    best_row_test_HJZ['smECE/agg'].values[0],
                    best_row_test_HJZ['smECE/max'].values[0],
                    best_row_test_HJZ['acc/agg'].values[0]]

        hjz_arr_std = [model + ' HJZ',
                          best_row_test_HJZ_std['ECE/agg'].values[0],
                          best_row_test_HJZ_std['ECE/max'].values[0],
                          best_row_test_HJZ_std['smECE/agg'].values[0],
                          best_row_test_HJZ_std['smECE/max'].values[0],
                          best_row_test_HJZ_std['acc/agg'].values[0]]
        
        # Get the best performing Platt row
        cf, id = get_best_param_id_cf(df_val, 'Platt', 'smECE/max', higher_is_better=False)
        best_row_test_platt = get_row_with_param_id(df_test, cf, 'HJZ', id)
        best_row_test_platt_std = get_row_with_param_id(df_std_test, cf, 'HJZ', id)
        platt_arr = [model + ' Platt',
                    best_row_test_platt['ECE/agg'].values[0],
                    best_row_test_platt['ECE/max'].values[0],
                    best_row_test_platt['smECE/agg'].values[0],
                    best_row_test_platt['smECE/max'].values[0],
                    best_row_test_platt['acc/agg'].values[0]]
        
        platt_arr_std = [model + ' Platt',
                        best_row_test_platt_std['ECE/agg'].values[0],
                        best_row_test_platt_std['ECE/max'].values[0],
                        best_row_test_platt_std['smECE/agg'].values[0],
                        best_row_test_platt_std['smECE/max'].values[0],
                        best_row_test_platt_std['acc/agg'].values[0]]

        # Get the best performing Temp row
        cf, id = get_best_param_id_cf(df_val, 'Temp', 'smECE/max', higher_is_better=False)
        best_row_test_temp = get_row_with_param_id(df_test, cf, 'Temp', id)
        best_row_test_temp_std = get_row_with_param_id(df_std_test, cf, 'Temp', id)
        temp_arr = [model + ' Temp',
                    best_row_test_temp['ECE/agg'].values[0],
                    best_row_test_temp['ECE/max'].values[0],
                    best_row_test_temp['smECE/agg'].values[0],
                    best_row_test_temp['smECE/max'].values[0],
                    best_row_test_temp['acc/agg'].values[0]]
        
        temp_arr_std = [model + ' Temp',
                        best_row_test_temp_std['ECE/agg'].values[0],
                        best_row_test_temp_std['ECE/max'].values[0],
                        best_row_test_temp_std['smECE/agg'].values[0],
                        best_row_test_temp_std['smECE/max'].values[0],
                        best_row_test_temp_std['acc/agg'].values[0]]
        

        # Get the best performing Isotonic row
        cf, id = get_best_param_id_cf(df_val, 'Isotonic', 'smECE/max', higher_is_better=False)
        best_row_test_isotonic = get_row_with_param_id(df_test, cf, 'Isotonic', id)
        best_row_test_isotonic_std = get_row_with_param_id(df_std_test, cf, 'Isotonic', id)
        isotonic_arr = [model + ' Isotonic',
                        best_row_test_isotonic['ECE/agg'].values[0],
                        best_row_test_isotonic['ECE/max'].values[0],
                        best_row_test_isotonic['smECE/agg'].values[0],
                        best_row_test_isotonic['smECE/max'].values[0],
                        best_row_test_isotonic['acc/agg'].values[0]]
        
        isotonic_arr_std = [model + ' Isotonic',
                            best_row_test_isotonic_std['ECE/agg'].values[0],
                            best_row_test_isotonic_std['ECE/max'].values[0],
                            best_row_test_isotonic_std['smECE/agg'].values[0],
                            best_row_test_isotonic_std['smECE/max'].values[0],
                            best_row_test_isotonic_std['acc/agg'].values[0]]

        # print(erm_arr)
        # print(hkrr_arr)
        # print(hjz_arr)

        # For each column except the first, find the smallest value and store it in a boolean array
        arr_best = [None] + [min(erm_arr[i], hkrr_arr[i], hjz_arr[i], platt_arr[i], temp_arr[i], isotonic_arr[i]) for i in range(1, 5)] + [max(erm_arr[5], hkrr_arr[5], hjz_arr[5], platt_arr[5], temp_arr[5], isotonic_arr[5])]

        # Convert to the appropriate strings
        erm_final = [model + ' ERM'] + [str_format(erm_arr[i], erm_arr_std[i], bold=(erm_arr[i] == arr_best[i])) for i in range(1, 6)]
        hkrr_final = [model + ' $\\hkrr$'] + [str_format(hkrr_arr[i], hkrr_arr_std[i], bold=(hkrr_arr[i] == arr_best[i])) for i in range(1, 6)]
        hjz_final = [model + ' $\\hjz$'] + [str_format(hjz_arr[i], hjz_arr_std[i], bold=(hjz_arr[i] == arr_best[i])) for i in range(1, 6)]
        platt_final = [model + ' Platt'] + [str_format(platt_arr[i], platt_arr_std[i], bold=(platt_arr[i] == arr_best[i])) for i in range(1, 6)]
        temp_final = [model + ' Temp'] + [str_format(temp_arr[i], temp_arr_std[i], bold=(temp_arr[i] == arr_best[i])) for i in range(1, 6)]
        isotonic_final = [model + ' Isotonic'] + [str_format(isotonic_arr[i], isotonic_arr_std[i], bold=(isotonic_arr[i] == arr_best[i])) for i in range(1, 6)]

        # Append the arrays to the result_df
        result_df = result_df._append(pd.Series(erm_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(hkrr_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(hjz_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(platt_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(temp_final, index=header), ignore_index=True)
        result_df = result_df._append(pd.Series(isotonic_final, index=header), ignore_index=True)

    # Print the dataframe
    # print('DATAFRAME ======================')
    # print(result_df)

    # Save the datafram with to_latex
    result_df.to_latex(SAVE_DIR + f'{dataset}_table.tex', index=False, index_names=False)

    # Load the table as a string
    with open(SAVE_DIR + f'{dataset}_table.tex', 'r') as f:
        table_str = f.read()
    
    # After the 7th line, add a new line with only '\midrule' 
    table_str = table_str.split('\n')
    k = 0
    odd = -1
    for i in range(4, len(table_str) - 1, 1):
        # Prepend '\rowcolor{Wheat1}' to the line on alternating sets of 3  rows
        if odd == -1 and k < 6 and len(models) > 1:
            table_str[i] = '\\rowcolor{Wheat1} ' + table_str[i]
        # Between every set of 3 rows, add a '\midrule' to separate out the performance of different
        # base models
        if i > 6 and k == 6:
            table_str.insert(i, '\midrule')
            k = 0
            # flip odd
            odd *= -1
        else:
            k += 1

    # Save the string back into the file
    with open(SAVE_DIR + f'{dataset}_table.tex', 'w') as f:
        f.write('\n'.join(table_str))

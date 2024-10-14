import os
import sys
# move to parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset import Dataset

def groups_tables_latex(groups_collection='default'):
    '''
    Save latex table of group information. Informaion used here is
    different from groups_table or group_stats as it does 
    not need information about each split.
    '''
    dir = f'figures/groups/{groups_collection}'
    os.makedirs(dir, exist_ok=True)

    tabular = ['ACSIncome', 'BankMarketing', 'CreditDefault', 'MEPS', 'HMDA']
    # non_tabular = ['CelebA', 'Camelyon17', 'YelpPolarity', 'AmazonPolarity', 'CivilComments']
    non_tabular = []
    for dataset_name in tabular + non_tabular:
        print(f'********** {dataset_name} **********')
        dataset = Dataset(dataset_name, groups=groups_collection, verbose=False)
        gps_df = dataset.groups_info_df()

        # remove dashes and underscores from gps_df['group name']
        gps_df['group name'] = gps_df['group name'].str.replace('_', ' ')
        # replace <, >, <=, >=, = with latex symbols
        gps_df['group name'] = gps_df['group name'].str.replace('<=', '$\leq$')
        gps_df['group name'] = gps_df['group name'].str.replace('>=', '$\geq$')
        gps_df['group name'] = gps_df['group name'].str.replace('<', '$<$')
        gps_df['group name'] = gps_df['group name'].str.replace('>', '$>$')
        gps_df['group name'] = gps_df['group name'].str.replace('=', '$=$')

        # delete old file if it exists
        if os.path.exists(f'{dir}/{dataset_name}.tex'):
            os.remove(f'{dir}/{dataset_name}.tex')

        # write in latex, add divider after first row of table
        with open(f'{dir}/{dataset_name}.tex', 'w') as f:
            latex_table = gps_df.to_latex(index=False)
            lines = latex_table.split('\n')
            # Insert \midrule on 3rd row from bottom
            lines.insert(-4, '\\midrule')
            f.write('\n'.join(lines))

if __name__ == '__main__':
    groups_tables_latex('default')
    groups_tables_latex('alternate')
import pandas as pd
pd.options.display.width = None

my_stat = pd.read_csv('datasets/my_stat_1.csv')

my_stat['session_value'] = my_stat['session_value'].fillna(0)
pos_median = my_stat[(my_stat['n_users'] >= 0)]['n_users'].median()
my_stat.loc[my_stat['n_users'] < 0, 'n_users'] = pos_median
print(my_stat)
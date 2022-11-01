import pandas as pd
pd.options.display.width = None

my_stat = pd.read_csv('datasets/my_stat.csv')

my_stat = my_stat.rename(columns={
    'V1': 'session_value',
    'V2': 'group',
    'V3': 'time',
    'V4': 'n_users'
})
print(my_stat)

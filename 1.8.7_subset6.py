import pandas as pd
pd.options.display.width = None

my_stat = pd.read_csv('datasets/my_stat_1.csv')
mean_session_value_data = my_stat.groupby('group', as_index=False).agg({'session_value': 'mean'})\
    .rename(columns={'session_value': 'mean_session_value'})


print(mean_session_value_data)
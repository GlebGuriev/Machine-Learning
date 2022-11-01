import pandas as pd
pd.options.display.width = None

my_stat = pd.read_csv('datasets/my_stat.csv')
print(my_stat)
subset_1 = my_stat[['V1', 'V3']].head(10)
print(subset_1)
subset_2 = my_stat[['V2', 'V4']].drop(0).drop(4)
print(subset_2)

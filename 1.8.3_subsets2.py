import pandas as pd
pd.options.display.width = None

my_stat = pd.read_csv('datasets/my_stat.csv')
print(my_stat)
subset_1 = my_stat.query('(V1 > 0) & (V3 == "A")')
print(subset_1)
subset_2 = my_stat.query('(V2 != 10) | (V4 >= 1)')
print(subset_2)

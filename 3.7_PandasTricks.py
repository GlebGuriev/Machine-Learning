import pandas as pd
import numpy as np
from time import time

df = pd.read_csv('datasets/iris.csv')

start = time()
df.describe().loc['mean']
finish = time()
print(finish-start)

start = time()
df.apply(np.mean)
finish = time()
print(finish-start)

start = time()
df.apply('mean')
finish = time()
print(finish-start)

start = time()
df.mean(axis=0)
finish = time()
print(finish-start)

(DATEDIFF(days_delivery, DATEDIFF(date_step_end, date_step_beg)) > 0, 0, DATEDIFF(days_delivery, DATEDIFF(date_step_end, date_step_beg))
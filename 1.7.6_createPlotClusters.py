import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.width = None

df = pd.read_csv('datasets/dataset_209770_6.txt', sep=" ")
df.plot.scatter(x='x', y='y')
plt.show()
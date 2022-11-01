import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width = None

df = pd.read_csv('datasets/iris.csv', index_col=0)
print(df)
sns.violinplot(df['petal length'])

plt.show()

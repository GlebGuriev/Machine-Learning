import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.width = None

df = pd.read_csv('datasets/iris.csv', index_col=0)
print(df)
df.plot.kde(subplots=True)

plt.show()

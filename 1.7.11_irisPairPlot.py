import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.width = None

iris = pd.read_csv('datasets/iris.csv', index_col=0)
print(iris)
print(iris.corr())
sns.pairplot(iris, vars=iris.columns[:4], hue="species")
plt.show()

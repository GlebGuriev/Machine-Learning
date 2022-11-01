import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.width = None

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/genome_matrix.csv', index_col=0)
heatmap = sns.heatmap(data=df, cmap='viridis')
heatmap.xaxis.set_ticks_position('top')
heatmap.xaxis.set_tick_params(rotation=90)
plt.show()

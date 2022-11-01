import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.width = None

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
df['cnt'] = df['roles'].str.count(',')+1
df['cnt'].hist()
plt.show()
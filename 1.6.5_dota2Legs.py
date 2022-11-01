import pandas as pd
pd.options.display.width = None

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')
print(df.groupby('legs').size())

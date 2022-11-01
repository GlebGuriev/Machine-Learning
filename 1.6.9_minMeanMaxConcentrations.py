import pandas as pd
pd.options.display.width = None

df = pd.read_csv('http://stepik.org/media/attachments/course/4852/algae.csv')
print(df.groupby('genus')['alanin'].min(), df.groupby('genus')['alanin'].mean(), df.groupby('genus')['alanin'].max())
import pandas as pd
pd.options.display.width = None

df = pd.read_csv('https://stepik.org/media/attachments/course/4852/accountancy.csv')
print(df.groupby(['Type', 'Executor'])['Salary'].mean().unstack())

import pandas as pd
pd.options.display.width = None

sp = pd.read_csv('https://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')
print(sp.groupby('lunch').mean())
print(sp.groupby('lunch').var())


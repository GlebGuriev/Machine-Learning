import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
pd.options.display.width = None

data = pd.read_csv("datasets/heart-disease.csv")
X = data.drop('target', axis=1)
y = data['target']

np.random.seed(0)
rf = RandomForestClassifier(10, max_depth=5)

rf.fit(X, y)

imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()

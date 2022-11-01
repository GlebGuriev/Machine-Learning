import pandas as pd
pd.options.display.width = None

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('datasets/songs.csv')
X = train_data.drop(['song', 'year', 'genre', 'lyrics', 'artist'], axis=1)
y = train_data['artist']
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
precision = precision_score(y_test, predictions, average='micro')

print(precision)

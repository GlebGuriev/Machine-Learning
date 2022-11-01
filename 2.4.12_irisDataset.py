import pandas as pd
pd.options.display.width = None

from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_iris = pd.read_csv('datasets/train_iris.csv', index_col=0)
test_iris = pd.read_csv('datasets/test_iris.csv', index_col=0)

X_train = train_iris.drop('species', axis=1)
y_train = train_iris['species']
X_test = test_iris.drop('species', axis=1)
y_test = test_iris['species']

scores = pd.DataFrame()
np.random.seed(0)
for max_depth in range(1, 100):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    score = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score], 'test_score': [test_score]})
    scores = scores.append(score)

sns.lineplot(scores, x='max_depth', y='train_score')
sns.lineplot(scores, x='max_depth', y='test_score')
plt.show()
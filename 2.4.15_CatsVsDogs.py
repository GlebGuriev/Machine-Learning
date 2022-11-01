import pandas as pd
pd.options.display.width = None

from sklearn import tree
from sklearn.model_selection import train_test_split

train_cat = pd.read_csv('datasets/dogs_n_cats.csv')
test_cat = pd.read_json('datasets/dataset_209691_15.txt')
X = train_cat[['Гавкает', "Лазает по деревьям"]]
y = train_cat['Вид']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(train_score, test_score)

res = clf.predict(test_cat[['Гавкает', "Лазает по деревьям"]])
print(list(res).count('собачка'))

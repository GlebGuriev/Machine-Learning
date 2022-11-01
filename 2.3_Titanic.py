import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
pd.options.display.width = None

# считываем данные
titanic_data = pd.read_csv('datasets/train.csv')

# убираем лишние переменные, заполняем неизвестный возраст медианой, устанавливаем завис.и независ. переменные
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X = pd.get_dummies(X)
X = X.fillna(X['Age'].median())
y = titanic_data['Survived']

#разделяем датасет на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# сажаем дерево, критерий - энтропия, максимальная глубина = 3
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)

# тренируем на тренировочном датасете
clf.fit(X_train, y_train)

# вывод данных - score на тренировочном и тестовом датасете одинаковый - это хорошо
print(titanic_data)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
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

# сажаем дерево, критерий - энтропия, максимальная глубина = 4
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf.fit(X_train, y_train)
# средняя точность на кросс-валидации
mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()

# вывод данных - score на тренировочном и тестовом датасете одинаковый - это хорошо
print(titanic_data)
print(mean_cross_val_score)

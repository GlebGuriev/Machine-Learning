import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
pd.options.display.width = None

# считываем данные
titanic_data = pd.read_csv('datasets/train.csv')

# убираем лишние переменные, заполняем неизвестный возраст медианой, устанавливаем завис.и независ. переменные
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X = pd.get_dummies(X)
X = X.fillna(X['Age'].median())
y = titanic_data['Survived']

# разделяем датасет на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# сажаем дерево, критерий - энтропия, максимальная глубина = 4

clf = tree.DecisionTreeClassifier()

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1,30)
}
grid_search_cv_clf = GridSearchCV(clf, params, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_

y_pred = best_clf.predict(X_test)
print(precision_recall_fscore_support(y_test, y_pred))

y_pred_prob = best_clf.predict_proba(X_test)

new_pred = np.where(y_pred_prob[:, 1] > 0.8, 1, 0)
print(precision_recall_fscore_support(y_test, new_pred))

# график roc-кривая
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


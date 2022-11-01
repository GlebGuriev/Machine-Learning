import pandas as pd
pd.options.display.width = None

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

events_data = pd.read_csv('datasets/events_data_test.csv')
submissions_data = pd.read_csv('datasets/submission_data_test.csv')


# print(submissions_data)
print(passed_id)
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.seed(0))

print('Training...')

clf = RandomForestClassifier()
params = {
    'n_estimators': range(50, 200),
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 10),
    'min_samples_split': range(2, 100),
    'min_samples_leaf': range(1, 50)
}
search = RandomizedSearchCV(clf, params, n_iter=20, n_jobs=-1)
search.fit(X_train, y_train)
best_tree = search.best_estimator_
predict = best_tree.predict(X_test)

precision = precision_score(y_test, predict, average='micro')
recall = recall_score(y_test, predict, average='micro')
auc = roc_auc_score(y_test, predict, average='micro')

print(precision, recall, auc)'''


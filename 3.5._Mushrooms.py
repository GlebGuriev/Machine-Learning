import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import show
pd.options.display.width = None

data = pd.read_csv("datasets/training_mush.csv")
train = pd.read_csv('datasets/testing_mush.csv')
right_answers = pd.read_csv('datasets/testing_y_mush.csv')
X = data.drop('class', axis=1)
y = data['class']

rf = RandomForestClassifier(random_state=0)

params = {
    'n_estimators': range(10, 50, 10),
    'max_depth': range(1, 12, 2),
    'min_samples_leaf': range(1, 7),
    'min_samples_split': range(2, 9, 2)
}

search = GridSearchCV(rf, params, cv=3, n_jobs=-1)
search.fit(X, y)

best_clf = search.best_estimator_

predictions = list(best_clf.predict(train))
count = predictions.count(1)
print(count)

conf_matrix = confusion_matrix(right_answers, predictions)
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
show()
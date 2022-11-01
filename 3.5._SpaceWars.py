import pandas as pd
pd.options.display.width = None

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score

invasion_data = pd.read_csv('datasets/invasion.csv')
operative_data = pd.read_csv('datasets/operative_information.csv')

X = invasion_data.drop('class', axis=1)
y = invasion_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np.random.randint(0, 100))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
print(imp.sort_values('importance', ascending=False))

'''                 Training

predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')

print(accuracy, precision, recall)                      '''

predictions = list(rf.predict(operative_data))

predicted_classes = {
    'cruiser': predictions.count('cruiser'),
    'transport': predictions.count('transport'),
    'fighter': predictions.count('fighter')
}

print(predicted_classes)

'''print(invasion_data.head())
print(invasion_data.shape)
print(operative_data.head())
print(operative_data.shape)'''
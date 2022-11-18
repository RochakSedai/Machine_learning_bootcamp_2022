from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
import numpy as np 

iris_data = datasets.load_iris()

# print(iris_data.target.shape)
features = iris_data.data
target = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)

fit_model = model.fit(feature_train, target_train)
predictions = fit_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
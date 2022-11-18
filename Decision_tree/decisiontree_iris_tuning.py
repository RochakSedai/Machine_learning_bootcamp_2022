from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
import numpy as np 

iris_data = datasets.load_iris()

# print(iris_data.target.shape)
features = iris_data.data
target = iris_data.target

# with grid search you can find the optmal parameter 'parameter tuning'
param_grid = {
    'max_depth': np.arange(1, 10)
}

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

tree.fit(feature_train, target_train)

print('Best parameter with grid search:', tree.best_params_)

grid_predictions = tree.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
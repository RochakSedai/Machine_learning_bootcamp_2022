from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
import numpy as np 

cancer_data = datasets.load_breast_cancer()

features =  cancer_data.data 
labels = cancer_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.2)

model = DecisionTreeClassifier(max_depth=3)

predicted = cross_validate(model, features, labels, cv=10)
print(np.mean(predicted['test_score']))
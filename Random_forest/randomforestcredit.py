import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
import math

# reading csv file
creditData = pd.read_csv(r'D:\AI Full Course\udemy\machine_learning\LogisticRegression\credit_data.csv')

# print(creditData.head())
# print(creditData.describe())
# print(creditData.corr())

features = creditData[['income', 'age', 'loan']]
target = creditData.default

# machine learning handle array not data frames
X= np.array(features).reshape(-1,3)
y = np.array(target)

# 30% of the dataset is for testing and 70% datatest for training
#feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = RandomForestClassifier()
predicted = cross_validate(model, X, y, cv=10)

print(np.mean(predicted['test_score']))



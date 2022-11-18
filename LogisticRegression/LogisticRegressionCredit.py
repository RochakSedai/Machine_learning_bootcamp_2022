import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import math

# reading csv file
creditData = pd.read_csv('credit_data.csv')

# print(creditData.head())
# print(creditData.describe())
# print(creditData.corr())

features = creditData[['income', 'age', 'loan']]
target = creditData.default


# 30% of the dataset is for testing and 70% datatest for training
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fitt = model.fit(feature_train, target_train)



prediction = model.fitt.predict(feature_test)

print(confusion_matrix(target_test, prediction))
print(accuracy_score(target_test, prediction))
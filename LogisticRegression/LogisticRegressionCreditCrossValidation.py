import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# reading csv file
creditData = pd.read_csv('credit_data.csv')

print(creditData.head())
# print(creditData.describe())
# print(creditData.corr())

features = creditData[['income', 'age', 'loan']]
target = creditData.default


# machine learning handles arrays not data-frame
x = np.array(features).reshape(-1,3)
y = np.array(target)

model = LogisticRegression()
predicted = cross_validate(model, x, y, cv=5)

print(np.mean(predicted['test_score']))
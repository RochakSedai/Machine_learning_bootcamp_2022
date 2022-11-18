import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing


# reading data from dataset
data = pd.read_csv(r"D:\AI Full Course\udemy\machine_learning\LogisticRegression\credit_data.csv")


features = data[['income', 'age', 'loan']]
target = data.default


#machine learning algo. handle arrays not dataframe
X = np.array(features).reshape(-1,3)
y = np.array(target)

X = preprocessing.MinMaxScaler().fit_transform(X)

feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=32)
fit_model = model.fit(feature_train, target_train)
prediction = fit_model.predict(feature_test)

cross_valid_scores = []

for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())
print(cross_valid_scores)
print('Optimal k with cross-validation: ', np.argmax(cross_valid_scores))

print(confusion_matrix(target_test, prediction))
print(accuracy_score(target_test, prediction))

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data = fetch_olivetti_faces()

# there are 400 images -  10x40 (40 people - 1 person has 10 images)- 1 image = 64x64 pixels
features = olivetti_data.data
targets = olivetti_data.target

# fig, subplots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
# subplots = subplots.flatten()

# for unique_user_id in np.unique(targets):
#     image_index = unique_user_id * 8
#     subplots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
#     subplots[unique_user_id].set_xticks([])
#     subplots[unique_user_id].set_yticks([])
#     subplots[unique_user_id].set_title('Facen id %s' % unique_user_id)
    

# plt.suptitle('The dataset (40 people)')
# plt.show()

# split the original dataset into training and test

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets, random_state=0)

# lets try out to find the optimal number of eigen vectors PCA
pca = PCA(n_components=100, whiten=True)
# pca.fit(features)

# plt.figure(1, figsize=(12,8))
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.xlabel('Components')
# plt.ylabel('Explained variances')
# plt.show()

pca.fit(x_train)
x_pca = pca.fit_transform(features)


x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# after we find the optimal 100 pca numbers we can check the 'eigenfaces'
# 1 principle component has 4096 featues

# number_of_eigenfaces = len(pca.components_)
# eigen_faces = pca.components_.reshape((number_of_eigenfaces, 64, 64))


# fig, subplots = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
# subplots = subplots.flatten()

# for i in range(number_of_eigenfaces):
#     subplots[i].imshow(eigen_faces[i], cmap='gray')
#     subplots[i].set_xticks([])
#     subplots[i].set_yticks([])

# plt.suptitle('Eigen faces')
# plt.show()

# lets use the machine learning models
models = [('Logistic Regression', LogisticRegression()), ('Support Vector Machine', SVC()), ('Naive Bayes classifier', GaussianNB())]

for name, model in models:
    # classifier_model = model
    # classifier_model.fit(x_train_pca, y_train)
    
    # y_predicted = classifier_model.predict(x_test_pca)
    # print('Results with %s'% name)
    # print('Accuracy score is: %s' % (metrics.accuracy_score(y_test, y_predicted)))

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, x_pca, targets, cv=kfold)
    print('Mean of the cross-validation scores: %s' % cv_scores.mean())

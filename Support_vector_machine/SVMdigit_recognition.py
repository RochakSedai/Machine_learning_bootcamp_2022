import matplotlib.pyplot as plt 
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix, accuracy_score

digits  = datasets.load_digits()

# print(digits.images)

images_and_labels = list(zip(digits.images, digits.target))

# for index, (image, label) in enumerate(images_and_labels[:6]):
#     plt.subplot(2,3, index+1)
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Target: %i' % label)

# plt.show()

# to applyt a classifier on this data, we need to flatten the image: instead of 8x8 matrix 
# we have to use a one-D array with 64 items
data = digits.images.reshape((len(digits.images), -1))
# print(data)

classifier = svm.SVC(gamma=0.001)
train_test_split = int(len(digits.images)*0.75)
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

# now predict the value of the digit pn the 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])


print('Confusion matrix: \n%s' % confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))

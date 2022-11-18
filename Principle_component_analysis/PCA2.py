from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mnist_data = fetch_openml('mnist_784')

features = mnist_data.data 
targets = mnist_data.target 

print(features.shape)

train_img, test_img, train_lbl, test_lbl = train_test_split(features, targets, test_size=0.15)

scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# we keep 95% variance
pca = PCA(0.95)

pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print(train_img.shape)


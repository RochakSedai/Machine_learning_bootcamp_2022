import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x, y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=3)


# print(x)
# print(y)  # integer representation of center or cluster

plt.scatter(x[:, 0], x[:, 1], s=50)
# plt.show()
model =KMeans(3)
model.fit(x)
y_kmeans = model.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans,  s=50, cmap='rainbow')
plt.show()
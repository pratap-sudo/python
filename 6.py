# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Create dataset (2D points)
X = np.array([
    [1, 2], [2, 1], [3, 2],
    [8, 9], [9, 8], [10, 9],
    [5, 6], [6, 5]
])

# Step 3: Apply KMeans
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

# Step 4: Get cluster labels and centers
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids:\n", centroids)

# Step 5: Plot clusters
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)
plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

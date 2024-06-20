import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate some random data points
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 points in 2-dimensional space

# Number of clusters
k = 3

# Initialize KMeans object
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit KMeans clustering model to the data
kmeans.fit(X)

# Predict the cluster labels
labels = kmeans.labels_

# Get the coordinates of the cluster centers
centers = kmeans.cluster_centers_

# Plotting the clusters
plt.figure(figsize=(8, 6))

# Color map
colors = ['r', 'g', 'b']

for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=100, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

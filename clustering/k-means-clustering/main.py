import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Loading the dataset from a CSV file
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # Selecting only the relevant features (Annual Income and Spending Score)

# Calculating WCSS (Within-Cluster Sum of Square) for different numbers of clusters
wcss = []
for i in range(1, 11):
    cluster = KMeans(n_clusters=i, init='k-means++', random_state=42)
    cluster.fit(X)
    wcss.append(cluster.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center

# Plotting the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  # This plot helps to determine the optimal number of clusters

# Applying KMeans to the dataset with the optimal number of clusters
cluster = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = cluster.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color='indigo', s=100, label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color='blue', s=100, label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color='orange', s=100, label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color='purple', s=100, label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color='green', s=100, label='Cluster 5')
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], marker='o', color='red', s=200,
            label='Centroid')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

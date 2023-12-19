# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Loading the dataset from a CSV file
dataset = pd.read_csv('Mall_Customers.csv')

# Extracting relevant features for clustering
X = dataset.iloc[:, [3, 4]].values  # Selecting 'Annual Income' and 'Spending Score'

# Creating a dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()  # The dendrogram shows the hierarchical relationship between data points

# Performing Agglomerative Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)  # The 'fit_predict' method assigns each data point to a cluster

# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], color='red', s=100, label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], color='blue', s=100, label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], color='orange', s=100, label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], color='purple', s=100, label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], color='green', s=100, label='Cluster 5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()  # Each cluster is visualized in a different color

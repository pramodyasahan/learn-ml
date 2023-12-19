import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    cluster = KMeans(n_clusters=i, init='k-means++', random_state=42)
    cluster.fit(X)
    wcss.append(cluster.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

cluster = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = cluster.fit_predict(X)


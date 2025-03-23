import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generowanie danych
np.random.seed(42)
X1 = np.random.normal(2, 1, (50, 2))
X2 = np.random.normal(8, 1, (50, 2))
X = np.vstack((X1, X2))

# Algorytm k-means
kmeans = KMeans(n_clusters=2, max_iter=100, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Wizualizacja
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label="Dane")
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label="Centroidy")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.title("Grupowanie k-means")
plt.legend()
plt.show()
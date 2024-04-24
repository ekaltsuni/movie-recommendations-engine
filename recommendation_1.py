from pre_process_2 import R_dataframe
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances


R = R_dataframe()

# Dataframe where each row represents a user and each column represents an item
R = R.pivot_table(index='Users', columns='Items', values='Ratings', fill_value=0)

# Feature vectors for each user
feature_vectors = R.values

# Custom Jaccard coefficient computation function
def custom_jaccard_coefficient(u, v):
    intersection = np.intersect1d(u, v)
    union = np.union1d(u, v)
    coefficient = 1.0 - len(intersection) / len(union)
    return coefficient

# Set the number of clusters (L)
L = 3

# Initialize KMeans
kmeans = KMeans(n_clusters=L, init='k-means++')

# Compute the pairwise Jaccard distances between all pairs of users
pairwise_jaccard_distances = pairwise_distances(feature_vectors, metric=custom_jaccard_coefficient)

# Fit KMeans to the pairwise Jaccard distances
cluster_labels = kmeans.fit_predict(pairwise_jaccard_distances)


# Print the clusters
for cluster_idx in range(L):
    print("Cluster", cluster_idx, ":")
    cluster_users = np.where(cluster_labels == cluster_idx)[0]
    for user_idx in cluster_users:
        print("User", user_idx)
    print()
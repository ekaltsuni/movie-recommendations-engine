from pre_process_2 import R_dataframe
from sklearn.cluster import KMeans
import numpy as np

R = R_dataframe()

# Dataframe where each row represents a user and each column represents an item
R = R.pivot_table(index='Users', columns='Items', values='Ratings', fill_value=0)

# Feature vectors for each user
feature_vectors = R.values

# Custom Jaccard coefficient computation function
def custom_jaccard_coefficient(cluster_labels, cluster_i_indices, cluster_j_indices):
    intersection = np.intersect1d(cluster_i_indices, cluster_j_indices)
    union = np.union1d(cluster_i_indices, cluster_j_indices)
    coefficient = 1.0 - len(intersection) / len(union)
    return coefficient

# Set number of clusters
L = 5

# Create KMeans model
kmeans = KMeans(n_clusters=L)

# Fit KMeans model and get cluster labels
cluster_labels = kmeans.fit_predict(feature_vectors)

# Compute the Jaccard coefficients between clusters
jaccard_coefficients = np.zeros((L, L))

for i in range(L):
    for j in range(i + 1, L):
        # Get the indices of users belonging to cluster i and cluster j
        cluster_i_indices = np.where(cluster_labels == i)[0]
        cluster_j_indices = np.where(cluster_labels == j)[0]

        # Compute the Jaccard coefficient between cluster i and cluster j
        jaccard_coefficient = custom_jaccard_coefficient(cluster_labels, cluster_i_indices, cluster_j_indices)
        jaccard_coefficients[i][j] = jaccard_coefficient
        jaccard_coefficients[j][i] = jaccard_coefficient

# Display Jaccard coefficients
for i in range(L):
    for j in range(i + 1, L):
        print("Jaccard coefficient between cluster", i, "and cluster", j, ":", jaccard_coefficients[i][j])

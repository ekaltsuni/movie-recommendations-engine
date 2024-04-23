from pre_process_2 import R_dataframe
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

R = R_dataframe()

# Dataframe where each row represents a user and each column represents an item
R = R.pivot_table(index='Users', columns='Items', values='Ratings', fill_value=0)

# Feature vectors for each user
feature_vectors = R.values
# Set the number of clusters (L)
L = 3
def compute_binary_vector(feature_vectors):
    binary_vectors = np.where(feature_vectors > 0, 1, 0)
    return binary_vectors

binary_vectors = compute_binary_vector(feature_vectors)


#
# def reduce_vectors(vector):
#     nmf = NMF(n_components=1)
#     nmf.fit(vector)
#     reduced_vectors = nmf.transform(feature_vectors)
#     return reduced_vectors

#reduced_feature_vectors = reduce_vectors(feature_vectors)
#reduced_binary_vectors = reduce_vectors(binary_vectors)

# euclidean distance based on equation
def euclidean_distance(R_u, R_v, lambda_u, lambda_v):
    # Υπολογισμός της απόστασης με βάση τη συνθήκη λj(k)
    squared_diff = np.sum(((R_u - R_v) ** 2) * lambda_u * lambda_v)
    distance = np.sqrt(squared_diff)
    return distance


# Compute pairwise distances using the custom distance function
custom_distances = pairwise_distances(feature_vectors, metric=euclidean_distance, **{'lambda_u': binary_vectors, 'lambda_v': binary_vectors})

print(custom_distances)

# Check the shapes of feature_vectors and custom_distances
print("Feature vectors shape:", feature_vectors.shape)
print("Binary vectors shape:", binary_vectors.shape)
print("Custom distances shape:", custom_distances.shape)



#Flatten custom distances array to create a 1D array of weights
custom_distances_1d = custom_distances.flatten()
print("Custom distances 1d shape:", custom_distances_1d.shape)

#print("Reduced custom distances shape:", reduced_custom_distances.shape)


print("Feature vectors:", feature_vectors)
print("Binary vectors:", binary_vectors)
print("Custom distances:", custom_distances)

print("Custom distances 1d:", custom_distances_1d)

# Initialize Kmeans
kmeans = KMeans(n_clusters=L, init='k-means++')

# Fit kmeans to data with the custom distance
kmeans.fit(feature_vectors, sample_weight=custom_distances)

cluster_labels = kmeans.labels_

#R['Cluster'] = kmeans.labels_
#print(R)



# Visualize clusters for n_clusters = 3
plt.figure(figsize=(10, 6))
plt.scatter(feature_vectors[:, 0], feature_vectors[:, 1], c=cluster_labels.labels_, cmap='viridis', alpha=0.5)
plt.title('User Clusters (n_clusters = 3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
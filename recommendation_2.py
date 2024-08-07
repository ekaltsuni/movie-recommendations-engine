import warnings
warnings.filterwarnings("ignore", message="JAX on Mac ARM machines is experimental and minimally tested.")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import Input
from tensorflow.keras.models import load_model
from pre_process_2 import R_dataframe
from recommendation_1 import custom_jaccard_coefficient

# Load and preprocess data
R = R_dataframe()
R = R.pivot_table(index='Users', columns='Items', values='Ratings', fill_value=0)
feature_vectors = R.values

# Number of clusters
L = 3
# Number of nearest neighbors
k = 5

# Initialize KMeans
kmeans = KMeans(n_clusters=L, init='k-means++')
pairwise_jaccard_distances = pairwise_distances(feature_vectors, metric=custom_jaccard_coefficient)
cluster_labels = kmeans.fit_predict(pairwise_jaccard_distances)

# Train an MLP for each cluster
mlp_models = []

for cluster_idx in range(L):
    print(f"Training MLP for cluster {cluster_idx}...")

    # Get users in the current cluster
    cluster_users = np.where(cluster_labels == cluster_idx)[0]
    cluster_data = feature_vectors[cluster_users]

    # Find k nearest neighbors for each user in the cluster
    knn = NearestNeighbors(n_neighbors=k, metric=custom_jaccard_coefficient)
    knn.fit(cluster_data)
    neighbors = knn.kneighbors(cluster_data, return_distance=False)

    # Prepare training data
    X_train = []
    y_train = []

    for user_idx, user_neighbors in enumerate(neighbors):
        X_train.append(cluster_data[user_neighbors])
        y_train.append(cluster_data[user_idx])

    X_train = np.array(X_train).reshape(len(cluster_users), -1)
    y_train = np.array(y_train)

    # Build and train MLP
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    mlp_models.append(model)

# Save models
for idx, model in enumerate(mlp_models):
    model.save(f'mlp_cluster_{idx}.keras')

print("Training complete!")

# Load and display one of the saved models
model = load_model('mlp_cluster_0.keras')
model.summary()

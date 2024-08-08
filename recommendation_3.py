import warnings
warnings.filterwarnings("ignore", message="JAX on Mac ARM machines is experimental and minimally tested.")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
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
results = []

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
    X = []
    y = []

    for user_idx, user_neighbors in enumerate(neighbors):
        X.append(cluster_data[user_neighbors])
        y.append(cluster_data[user_idx])

    X = np.array(X).reshape(len(cluster_users), -1)
    y = np.array(y)

    # Horizontal concatenation of feature vectors (neighbors) and target vectors (user preferences)
    data = np.hstack((X, y))

    # Random split into training (90%) and testing (10%) sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Split into features and labels
    X_train = train_data[:, :-y.shape[1]]
    y_train = train_data[:, -y.shape[1]:]
    X_test = test_data[:, :-y.shape[1]]
    y_test = test_data[:, -y.shape[1]:]

    # Build and train MLP
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    mlp_models.append(model)

    # Evaluate the model using MAE
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Store the results
    results.append({
        'Cluster': cluster_idx,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    })

# Print the results
results_df = pd.DataFrame(results)
print(results_df)

# Save models
for idx, model in enumerate(mlp_models):
    model.save(f'mlp_cluster_{idx}.keras')

print("Training complete!")

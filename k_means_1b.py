#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from dataset import load_dataframe
from scipy.sparse import csr_matrix 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the DataFrame
X = load_dataframe()


# Splitting the single column into separate columns
X = X[0].str.split(',', expand=True)
X = X.iloc[:, 0:3]
X.columns = ['Users', 'Items', 'Ratings']

# Dropping possible duplicates
X = X.drop_duplicates()


# Convert to numeric values
X['Ratings'] = pd.to_numeric(X['Ratings'])
X['Users'] = X['Users'].str.extract('(\d+)').astype(int)
X['Items'] = X['Items'].str.extract('(\d+)').astype(int)





#map each user and each item to array index
def mapData(S):
    #Get unique items
    unique_itemSet=set(S['Items'].unique())
    sortedItems=sorted(unique_itemSet)
    
    #Get unique users
    unique_userSet=set(S['Users'].unique())
    sortedUsers=sorted(unique_userSet)
    
    #map each item to array index
    itemToIndex=dict()
    for i in range(len(sortedItems)):
        itemToIndex[sortedItems[i]]=i
    
    #map each user to array index
    userToIndex=dict()
    for i in range(len(sortedUsers)):
        userToIndex[sortedUsers[i]]=i
    return [unique_itemSet,itemToIndex,unique_userSet,userToIndex]



[unique_itemSet,itemToIndex,unique_userSet,userToIndex]=mapData(X)




 
#Return the constrained dataset of those users having more than minR and less than maxR number of reviews
def R_dataframe(minR,maxR,X):
    user_counts = X.iloc[:, 0].value_counts()
    filtered_users = user_counts[(user_counts >= minR) & (user_counts < maxR)]
    constrainedX = X[X.iloc[:, 0].isin(filtered_users.index)]
    return constrainedX


#Create an empty sparse array with int8 elements
R = csr_matrix((len(unique_userSet), len(itemToIndex)),
                          dtype = np.int8).toarray()

#Get those records of the users within Rmin and Rmax
constrainedX=R_dataframe(69,73,X)

#create index for the constrained dataset
[constrained_unique_itemSet,constrained_itemToIndex,constrained_unique_userSet,constrained_userToIndex]=mapData(constrainedX)

#Create an empty sparse array with int8 elements
constrainedR = csr_matrix((len(constrained_unique_userSet), len(constrained_itemToIndex)),
                          dtype = np.int8).toarray()

#Fill in the R for the constrained dataset - sparse array
for i in range(len(constrainedX)):
    if i%100000==0:
      print('{0:.0%}'.format(i/len(constrainedX)))
    constrainedR[constrained_userToIndex[constrainedX.iloc[i]["Users"]]][constrained_itemToIndex[constrainedX.iloc[i]["Items"]]]=constrainedX.iloc[i]["Ratings"]


    
def distCosine(u,v):
    #m=len(constrainedR[u])
    m=len(u)
    numerator=0
    denominatorTemp1=0
    denominatorTemp2=0
    for k in range(m):
        lu=1 if u[k]>0 else 0
        lk=1 if v[k]>0 else 0
        temp1=u[k]
        temp2=v[k]
        numerator=numerator+temp1*temp2*lu*lk
        denominatorTemp1=denominatorTemp1+temp1*temp1*lu*lk
        denominatorTemp2=denominatorTemp2+temp2*temp2*lk*lk
    denominatorTemp1=math.sqrt(denominatorTemp1)
    denominatorTemp2=math.sqrt(denominatorTemp2)
    
    if denominatorTemp1==0 or denominatorTemp2==0 :
        #print("user "+str(u)+" and user "+str(v)+" have no common movie")
        return 1
                
    return 1-(abs(numerator/(denominatorTemp1*denominatorTemp2)))


# Initialize K-Means
def kmeans_custom_distance(data, k, distance_func, max_iterations=10):
    # Randomly initialize centroids
    num_samples, num_features = data.shape
    #centroids = data[np.random.choice(num_samples, k, replace=False)]
    centroids = np.random.uniform(0, 10, size=(k, num_features))
    # To store the cluster assignments
    cluster_assignments = np.zeros(num_samples)

    for iteration in range(max_iterations):
        print("Iteration: "+str(iteration))
        # Step 1: Assign points to the nearest centroid using the custom distance function
        for i in range(num_samples):
            distances = [distance_func(data[i], centroid) for centroid in centroids]
            cluster_assignments[i] = np.argmin(distances)
        
        # Step 2: Recalculate the centroids
        new_centroids = np.zeros((k, num_features))
        for cluster_idx in range(k):
            points_in_cluster = data[cluster_assignments == cluster_idx]
            if len(points_in_cluster) > 0:
                new_centroids[cluster_idx] = np.mean(points_in_cluster, axis=0)
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, cluster_assignments


def plot_clusters_pca(data, cluster_assignments, centroids):
    # Apply PCA to reduce data to 2D for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Reduce the centroids to 2D using the same PCA model
    reduced_centroids = pca.transform(centroids)
    
    # Plot the data points with their cluster assignments
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap='viridis', marker='o', alpha=0.7)
    
    # Plot the centroids
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    
    plt.title('Cluster Visualization with PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    



#y=stats()
k=3 
centroids, cluster_assignments = kmeans_custom_distance(constrainedR, k, distCosine)

plot_clusters_pca(constrainedR, cluster_assignments, centroids)


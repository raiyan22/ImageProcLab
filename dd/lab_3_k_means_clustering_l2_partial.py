# -*- coding: utf-8 -*-
"""LAB-3-K-Means-Clustering-L2-partial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M42FHepFWSNiqrg4SlvwHS_aZB36cfTY
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Make_blobs dataset for clustering.
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=250, centers=5, random_state=42)
# Number of training examples and cluster centers.
m = X.shape[0] 
n = X.shape[1] 
n_iter = 100

# Shape of the dataset.
print(X.shape, y.shape)
print(m,n)

# Inspect the dataset.
print(X[0:4])
y[0:4]

# Plot the clusters.
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(X[:,0],X[:,1])
plt.title('Original Dataset')

# Compute the initial centroids randomly.
import random
K=5
# Create an empty centroid array.
centroids = np.array([]).reshape(n,0) #shape = (2,5)
# Create 5 random centroids.
for k in range(K):
    centroids = np.c_[centroids, X[random.randint(0,m-1)]] #randint(0,249)

print(centroids[0:4])
centroids.shape

# Create an empty array.
euclid = np.array([]).reshape(m,0)
# Find distance between from each points to three centroids.
for k in range(K):
       dist = np.sum((X-centroids[:,k])**2, axis=1) # dist = (250,)
       euclid = np.c_[euclid, dist] 
# euclid = (250,5)
# Store the minimum value we have computed.
minimum = np.argmin(euclid, axis=1)+1
minimum

# Repeat the steps 
for i in range(n_iter):
    #Calculate the distance.
    #Write your code here for distance calculation.
    
    cent = {}
    for k in range(K):
        cent[k+1] = np.array([]).reshape(0,2)

    # Assign clusters to points.
    for k in range(m):
        #write your code here and remove continue.
        continue 

    # Compute mean and update.
    for k in range(K):
        #write your code here and remove continue.
        continue 

    final = cent

plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(X[:,0], X[:,1])
plt.title('Original Dataset')

plt.figure(figsize=(10, 7), dpi=100)
for k in range(K):
    plt.scatter(final[k+1][:,0], final[k+1][:,1])
plt.scatter(centroids[0,:], centroids[1,:], s=100, c='purple')
plt.show()

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, centers=3, random_state=42)

import seaborn as sns
from sklearn.cluster import KMeans
elbow=[]
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    elbow.append(kmeans.inertia_)
    print('{}: {}'.format(i, elbow[i-1]))
plt.figure(figsize=(10, 7), dpi=100)
sns.lineplot(range(1, 30), elbow,color='red')
plt.title('Elbow Method')
plt.show()
# PRODIGY_ML_02

# Customer Segmentation using K-Means Clustering

This repository contains code for performing customer segmentation using the K-Means clustering algorithm. The goal is to segment customers based on their annual income and spending score.

## Dataset

The dataset used in this project is `Mall_Customers.csv`. The dataset includes the following features:

- `CustomerID`: Unique ID for each customer.
- `Gender`: Gender of the customer.
- `Age`: Age of the customer.
- `Annual Income (k$)`: Annual income of the customer in thousands of dollars.
- `Spending Score (1-100)`: Spending score assigned to the customer (1-100).

## Installation

To run this code, you need to have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install these libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib


Ensure Mall_Customers.csv is in the same directory as the code.
Run the script to load the dataset, preprocess the data, apply K-Means clustering, and visualize the results.

Explanation of code:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv('Mall_Customers.csv')

# Selecting features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-Means clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Adding cluster labels to the dataset
data['Cluster'] = labels

# Visualizing the clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='x', color='black', label='Centroids', s=100)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Customers')
plt.legend()
plt.show()

Explanation of Key StepsData Loading: 
The dataset is loaded using pandas and checked for any missing values.

Feature Selection: 
The features used for clustering are Annual Income (k$) and Spending Score (1-100).

Data Standardization: 
The features are standardized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 
1.K-Means Clustering: 
The KMeans algorithm from scikit-learn is used to cluster the customers into 5 clusters (k=5).

Visualization: 
The clusters are visualized using matplotlib with different colors for each cluster and centroids marked with black 'x' symbols.
ContributingFeel free to fork this repository and make modifications. Pull requests are welcome.
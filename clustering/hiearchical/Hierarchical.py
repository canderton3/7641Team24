import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

def load_data():
    w_score_raw = 'https://raw.githubusercontent.com/canderton3/7641Team24/master/data/exoplanet_cleanedrf_w_score.csv'
    no_score_raw = 'https://raw.githubusercontent.com/canderton3/7641Team24/master/data/exoplanet_cleanedrf.csv'
    w_score = pd.read_csv(w_score_raw).to_numpy()
    no_score = pd.read_csv(no_score_raw).to_numpy()
    # Get Labels
    labels = no_score[:, 0]
    # Delete Labels from data
    w_score = np.delete(w_score, 0, axis=1)
    no_score = np.delete(no_score, 0, axis=1)
    return w_score, no_score, labels

w_score = load_data()[0]
no_score = load_data()[1]
labels = load_data()[2]

def scale_data(X):
    binary_vars = X[:, :4]
    print(binary_vars)
    # Scale non-binary variables
    scaled_numeric = StandardScaler().fit_transform(X[: , 4:])
    # Recombine
    X = np.concatenate((binary_vars, scaled_numeric), axis = 1)
    return X



scaled_no_score = scale_data(no_score)

#Build dendrogram
dendrogram = sch.dendrogram(sch.linkage(scaled_no_score, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Data Point')
plt.ylabel('Euclidean Distances')
plt.show()

#Visually inspect, create 5 clusters given created dendrogram
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')

hier_labels = hc.fit_predict(scaled_no_score)

#Build PCA for plotting
ndimensions = 2

pca = PCA(n_components=ndimensions)
pca.fit(scaled_no_score)
X_pca_array = pca.transform(scaled_no_score)

plt.scatter(X_pca_array[:,0], X_pca_array[:,1], c=hier_labels)

# Find percent candidate / false positive in each cluster
for i in range (max(hier_labels) + 1):
    points_in_cluster = scaled_no_score[np.where(hier_labels == i)]
    points_in_cluster_label = labels[np.where(hier_labels == i)]
    (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("CLUSTER ", i)
    print(frequencies)


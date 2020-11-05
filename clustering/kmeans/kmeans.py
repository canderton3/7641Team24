import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
import importlib, importlib.util
from kneed import KneeLocator


# def module_from_file(module_name, file_path):
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# cc = module_from_file("ClusterClass", "../ClusterClass/cluster_class.py")



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



kmeans_kwargs = {
    'init' : "random",
    'n_clusters' : 15,
    'n_init' : 10,
    'max_iter' : 300,
    'random_state' :42,
}

# kmeans = KMeans(**kmeans_kwargs)

# k_labels = kmeans.fit(scaled_no_score)

# print(k_labels)
# A list holds the SSE values for each k
sse = []
for k in range(1,16):
    kmeans = KMeans(k, **kmeans_kwargs)
    kmeans.fit(scaled_no_score)
    sse.append(kmeans.inertia_)

print(sse)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 16), sse)
plt.xticks(range(1, 16))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


kl = KneeLocator(
    range(1, 16), sse, curve="convex", direction="decreasing"
)

print(kl.elbow)

 # A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 16):
    kmeans = KMeans(k, **kmeans_kwargs)
    kmeans.fit(scaled_no_score)
    score = silhouette_score(scaled_no_score, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 16), silhouette_coefficients)
plt.xticks(range(2, 16))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

print(silhouette_coefficients)

kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
    )

k_labels = kmeans.fit_predict(scaled_no_score)
#plt.scatter(k_labels[0][:,0],k_labels[0][:,1],c=k_labels,cmap='brg')


def plot_dbscan(X, y, dbscan_labels):
    # Get two most important factors to plot
    pca = PCA(n_components=2)
    x = pca.fit_transform(X)
    # Plot each cluster
    num_clusters = max(dbscan_labels) + 1
    for i in range(num_clusters):
        points_in_cluster = x[np.where(dbscan_labels == i)]
        labels = y[np.where(dbscan_labels == i)]
        rgb = (np.random.random(), np.random.random(), np.random.random())
        for j in range(points_in_cluster.shape[0]):
            if (labels[j] == "CANDIDATE"):  
                plt.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='*')
            else:
                plt.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker=',')
        print("CLUSTER ", i)
    print("finished")
    plt.show()

plot_dbscan(scaled_no_score, labels, k_labels)


pca = PCA(2)
pca.fit(scaled_no_score)
X_pca_array = pca.transform(scaled_no_score)

plt.scatter(X_pca_array[:,0], X_pca_array[:,1], c=k_labels)
# print(k_labels.labels_[:5])

# plt.scatter(k_labels[:, 0], k_labels[:, 1], c=k_labels, s=50, cmap='viridis')

# centers = k_labels.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


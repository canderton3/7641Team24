import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#sklearn imports
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn import metrics

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def load_data():
    w_score = pd.read_csv(r"../../data/exoplanet_cleanedrf_w_score.csv").to_numpy()
    no_score = pd.read_csv(r"../../data/exoplanet_cleanedrf.csv").to_numpy()
    # Get Labels
    labels = no_score[:, 0]
    # Delete Labels from data
    w_score = np.delete(w_score, 0, axis=1)
    no_score = np.delete(no_score, 0, axis=1)
    return w_score, no_score, labels

def dbscan(X, y):
    binary_vars = X[:, :4]
    print(binary_vars)
    # Scale non-binary variables
    scaled_numeric = StandardScaler().fit_transform(X[: , 4:])
    # Recombine
    X = np.concatenate((binary_vars, scaled_numeric), axis = 1)

    # Compute DBSCAN
    db = DBSCAN(eps=0.7, min_samples=15).fit(X)
    labels = db.labels_
    print("NEW LABELS: ", max(labels))
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(y, labels))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels))

    # Find percent candidate / false positive in each cluster
    for i in range (max(labels) + 1):
        points_in_cluster = X[np.where(labels == i)]
        points_in_cluster_label = y[np.where(labels == i)]
        (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print("CLUSTER ", i)
        print(frequencies)


    #plot_dbscan(X=X, y=y, dbscan_labels=labels)


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

if __name__ == "__main__":
    w_score, no_score, labels = load_data()
    print(labels)
    dbscan(no_score, labels)


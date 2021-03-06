import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#sklearn imports
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn import metrics


def load_data():
    top_5 = pd.read_csv(r"../../data/exoplanet_cleanedrf_top_5.csv").to_numpy()
    top_10 = pd.read_csv(r"../../data/exoplanet_cleanedrf_top_10.csv").to_numpy()
    # Get Labels
    labels = top_5[:, 0]
    # Delete Labels from data
    top_10 = np.delete(top_10, 0, axis=1)
    top_5 = np.delete(top_5, 0, axis=1)
    return top_10, top_5, labels

def dbscan(X, y):
    X_df = X
    binary_vars = X[:, :4]
    print(binary_vars)
    # Scale non-binary variables
    scaled_numeric = StandardScaler().fit_transform(X[: , 4:])
    # Recombine
    X = np.concatenate((binary_vars, scaled_numeric), axis = 1)

    flag = True
    sils = []
    ep = 0.5
    min_samples = 6
    # Compute DBSCAN
    db = DBSCAN(eps=ep, min_samples=min_samples).fit(X)
    labels = db.labels_
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
 
    # Find percent candidate / false positive in each cluster
    for i in range (max(labels) + 1):
        points_in_cluster = X[np.where(labels == i)]
        points_in_cluster_label = y[np.where(labels == i)]
        (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print("CLUSTER ", i)
        print(frequencies)
    # Assess first cluster data
    cluster_0 = X_df[np.where(labels == 1)]
    assess_candidate_points(cluster_0)
    #plot_dbscan(X=X, y=y, dbscan_labels=labels)
    return labels


def plot_dbscan(X, y, dbscan_labels):
    # Get two most important factors to plot
    pca = PCA(n_components=2)
    x = pca.fit_transform(X)
    # Plot each cluster
    fig = plt.figure()
    ax = plt.axes()

    num_clusters = max(dbscan_labels) + 1
    for i in range(num_clusters):
        points_in_cluster = x[np.where(dbscan_labels == i)]
        labels = y[np.where(dbscan_labels == i)]
        rgb = (np.random.random(), np.random.random(), np.random.random())
        for j in range(points_in_cluster.shape[0]):
            if (labels[j] == "CANDIDATE"):  
                ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='*')
            else:
                ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='.')
        print("CLUSTER ", i)
    ax.set_title("DBSCAN Clusters, First Two PCA Components")
    ax.set_xlabel("PCA Component 2")
    ax.set_ylabel("PCA Component 1")
    plt.show()

def assess_candidate_points(cluster_0):
    print(cluster_0)
    df = pd.DataFrame(data=cluster_0, columns=["koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad"])
    # Averages
    print("AVERAGES\n koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}\n koi_prad: {}".format(
        df["koi_fpflag_co"].mean(), 
        df["koi_fpflag_nt"].mean(), 
        df["koi_fpflag_ss"].mean(),
        df["koi_fpflag_ec"].mean(),
        df["koi_prad"].mean()))
    # Modes for Binary
    print("MODES\n koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}".format(
        df["koi_fpflag_co"].mode(), 
        df["koi_fpflag_nt"].mode(), 
        df["koi_fpflag_ss"].mode(),
        df["koi_fpflag_ec"].mode()))

    print("COUNT ZEROS: koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}".format(
        df["koi_fpflag_co"].value_counts(), 
        df["koi_fpflag_nt"].value_counts(), 
        df["koi_fpflag_ss"].value_counts(),
        df["koi_fpflag_ec"].value_counts()))


if __name__ == "__main__":
    top_10, top_5, labels = load_data()

    dbscan_labels = dbscan(top_5, labels)
    data_w_labels = np.concatenate((top_5, dbscan_labels.reshape(dbscan_labels.shape[0], 1)), axis=1)
    data_w_labels = np.concatenate((labels.reshape(labels.shape[0], 1), data_w_labels), axis=1)

    clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
    clustered_df.to_csv("../../data/exoplanet_clustered_top_5.csv", index=False)



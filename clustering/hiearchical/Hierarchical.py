import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

def load_data():
    top_five_raw = 'https://raw.githubusercontent.com/canderton3/7641Team24/master/data/exoplanet_cleanedrf_top_5.csv'
    top_ten_raw = 'https://raw.githubusercontent.com/canderton3/7641Team24/master/data/exoplanet_cleanedrf_top_10.csv'
    top_five = pd.read_csv(top_five_raw).to_numpy()
    top_ten = pd.read_csv(top_ten_raw).to_numpy()
    # Get Labels
    labels = top_five[:, 0]
    # Delete Labels from data
    top_five = np.delete(top_five, 0, axis=1)
    top_ten = np.delete(top_ten, 0, axis=1)
    return top_five, top_ten, labels

top_five = load_data()[0]
top_ten = load_data()[1]
labels = load_data()[2]

def scale_data(X):
    binary_vars = X[:, :4]
    print(binary_vars)
    # Scale non-binary variables
    scaled_numeric = StandardScaler().fit_transform(X[: , 4:])
    # Recombine
    X = np.concatenate((binary_vars, scaled_numeric), axis = 1)
    return X

scaled_top_five = scale_data(top_five)
scaled_top_ten = scale_data(top_ten)

#Build dendrogram
def build_dendrogram(X):
    sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Data Point')
    plt.ylabel('Euclidean Distances')
    plt.show()
    
build_dendrogram(scaled_top_five)
build_dendrogram(scaled_top_ten)

#Create clustering, check silhouette scores
def Hierarchical(X):
    flag = True
    sils = []
    cluster = 6
    while (flag):
        # Compute Hierarchical Clustering
        hc = AgglomerativeClustering(n_clusters=cluster, affinity = 'euclidean', linkage = 'ward')
        hier_labels = hc.fit_predict(X)
        sils.append(metrics.silhouette_score(X, hier_labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, hier_labels))
        cluster += 1
        if (cluster > 25):
            break

    # Plot Silhouette Coefficients
    clusters = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    sil_clusters = plt.plot(clusters, sils)
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.show()
    
    for i, j in enumerate(sils):
        if j == max(sils):
            max_cluster = i+6
    return max_cluster

optimal_clusters_five = Hierarchical(scaled_top_five)
optimal_clusters_ten = Hierarchical(scaled_top_ten)

print(optimal_clusters_five, optimal_clusters_ten)


#build final cluster based on optimal number of clusters for top five and top ten
hc_five = AgglomerativeClustering(n_clusters=optimal_clusters_five, affinity = 'euclidean', linkage = 'ward')
hier_labels_five = hc_five.fit_predict(scaled_top_five)

hc_ten = AgglomerativeClustering(n_clusters=optimal_clusters_ten, affinity = 'euclidean', linkage = 'ward')
hier_labels_ten = hc_ten.fit_predict(scaled_top_ten)


#Plot optimal clustering on PCA for top five and top ten dataset
def plot_Hierarchical(X, y, hierarchical_labels):
    # Get two most important factors to plot
    pca = PCA(n_components=2)
    x = pca.fit_transform(X)
    # Plot each cluster
    fig = plt.figure()
    ax = plt.axes()

    num_clusters = max(hierarchical_labels) + 1
    for i in range(num_clusters):
        points_in_cluster = x[np.where(hierarchical_labels == i)]
        labels = y[np.where(hierarchical_labels == i)]
        rgb = (np.random.random(), np.random.random(), np.random.random())
        for j in range(points_in_cluster.shape[0]):
            if (labels[j] == "CANDIDATE"):  
                ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='*')
            else:
                ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='.')
        print("CLUSTER ", i)
    ax.set_title("Hierarchical Clusters, First Two PCA Components")
    ax.set_xlabel("PCA Component 2")
    ax.set_ylabel("PCA Component 1")
    plt.show()
    
plot_Hierarchical(scaled_top_ten, labels, hier_labels_ten)
plot_Hierarchical(scaled_top_five, labels, hier_labels_five)


# Find percent candidate / false positive in each cluster
def find_difference(X, my_labels):
    for i in range (max(my_labels) + 1):
        points_in_cluster = X[np.where(my_labels == i)]
        points_in_cluster_label = labels[np.where(my_labels == i)]
        (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print("CLUSTER ", i)
        print(frequencies)
    
find_difference(scaled_top_five, hier_labels_five)
find_difference(scaled_top_ten, hier_labels_ten)



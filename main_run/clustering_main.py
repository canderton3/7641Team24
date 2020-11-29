import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
#sklearn imports
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding

from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn import metrics
from kneed import KneeLocator


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

from clustering.gaussian_mixture_modeling.GMM import n_components_range, bic, spl, bars, cv_types
from main_run.clustering_run import plot_label



class MainMethods():
    def load_scaled_data(self, file):
        # read in file
        X= pd.read_csv(file).to_numpy()
        # Get Labels
        top_n_labels = X[:, 0]
        # Delete Labels from data
        X = np.delete(X, 0, axis=1)

        binary_vars = X[:, :4]
        # Scale non-binary variables
        scaled_numeric = StandardScaler().fit_transform(X[:, 4:])
        # Recombine
        scaled_top_n = np.concatenate((binary_vars, scaled_numeric), axis=1)
        return scaled_top_n, top_n_labels

        # Find percent candidate / false positive in each cluster

    def candidate_check(self, X, y, cluster_labels):
        candidate_count = {}
        for i in range(max(cluster_labels) + 1):
            points_in_cluster_label = y[np.where(cluster_labels == i)]
            (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print("CLUSTER ", i)
            print(frequencies)
            candidate_count[i] = frequencies
        # Assess first cluster data
        cluster_0 = X[np.where(cluster_labels == 1)]
        self.assess_candidate_points(cluster_0)
        # plot_dbscan(X=X, y=y, dbscan_labels=labels)
        return candidate_count

    def assess_candidate_points(self, cluster_0):
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

    def plot_pca(self, X, y, cluster_labels):
        # Get two most important factors to plot
        pca = PCA(n_components=2)
        x = pca.fit_transform(X)
        # Plot each cluster
        fig = plt.figure()
        ax = plt.axes()

        num_clusters = max(cluster_labels) + 1
        for i in range(num_clusters):
            points_in_cluster = x[np.where(cluster_labels == i)]
            labels = y[np.where(cluster_labels == i)]
            rgb = (np.random.random(), np.random.random(), np.random.random())
            for j in range(points_in_cluster.shape[0]):
                if labels[j] == "CANDIDATE":
                    ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='*')
                else:
                    ax.scatter(points_in_cluster[j, 0], points_in_cluster[j, 1], c=[rgb], marker='.')
            print("CLUSTER ", i)
        ax.set_title("DBSCAN Clusters, First Two PCA Components")
        ax.set_xlabel("PCA Component 2")
        ax.set_ylabel("PCA Component 1")
        plt.show()

    def build_dendrogram(self, X):
        sch.dendrogram(sch.linkage(X, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Data Point')
        plt.ylabel('Euclidean Distances')
        plt.show()

class ClusterModels:
    #dbscan model scaled data: X
    def dbscan_model(self, X):
        ep = 0.5
        min_samples = 6
        # Compute DBSCAN
        db = DBSCAN(eps=ep, min_samples=min_samples).fit(X)
        cluster_labels = db.labels_
        sc = metrics.silhouette_score(X, cluster_labels)
        print("Silhouette Coefficient: %0.3f" % sc)
        return cluster_labels, sc

    def kmeans_model(self, X, plot_label):
        # set parameters
        kmeans_kwargs = {
            'init': "random",
            'n_clusters': 15,
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42,
        }

        # A list holds the SSE values for each k for elbow curve
        sse = []
        for k in range(1, kmeans_kwargs['n_clusters']+1):
            kmeans = KMeans(k, **kmeans_kwargs)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)
        if plot_label:
            plt.style.use("fivethirtyeight")
            plt.plot(range(1, kmeans_kwargs['n_clusters']+1), sse)
            plt.xticks(range(1, kmeans_kwargs['n_clusters']+1))
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
            plt.show()
        else:
            pass
        #find elbow in elbow curve
        kl = KneeLocator(range(1, kmeans_kwargs['n_clusters']+1), sse, curve="convex", direction="decreasing")
        knee = kl.elbow
        #print("knee ", knee)
        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []
        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, kmeans_kwargs['n_clusters']+1):
            kmeans = KMeans(k, **kmeans_kwargs)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_coefficients.append(score)

        if plot_label:
            plt.style.use("fivethirtyeight")
            plt.plot(range(2, kmeans_kwargs['n_clusters']+1), silhouette_coefficients)
            plt.xticks(range(2, kmeans_kwargs['n_clusters']+1))
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.show()
        else:
            pass

        print(silhouette_coefficients)

        kmeans = KMeans(
            init="random",
            n_clusters=knee,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        k_labels = kmeans.fit_predict(X)

        return k_labels, silhouette_coefficients

    # Create clustering, check silhouette scores
    def Hierarchical(self, X):
        flag = True
        sils = []
        cluster = 6
        while (flag):
            # Compute Hierarchical Clustering
            hc = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='ward')
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
                max_cluster = i + 6
        return  hier_labels, sils

    def gmm_model(self, X, plot_label):
        # Determining number of components using BIC
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 10)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    low_gmm = gmm

        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])
        clf = low_gmm
        bars = []
        # Plot the BIC scores for each covariance type (star the lowest score)
            plt.figure(figsize=(8, 6))
            spl = plt.subplot(2, 1, 1)
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + .2 * (i - 2)
                bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                              (i + 1) * len(n_components_range)],
                                    width=.2, color=color))
        if plot_label:
            plt.xticks(n_components_range)
            plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
            plt.title('BIC score per model')
            xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
                   .2 * np.floor(bic.argmin() / len(n_components_range))
            plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
            spl.set_xlabel('Number of components')
            spl.legend([b[0] for b in bars], cv_types)
        else:
            pass
        # Select 2 as the number of components and build final GMM model
        final_gmm = GaussianMixture(n_components=2)
        final_gmm.fit(X)
        labels_cluster_gmm = gmm.predict(X)

        # Build PCA for plotting
        # ndimensions = 2
        #
        # pca = PCA(n_components=ndimensions)
        # pca.fit(X)
        # X_pca_array = pca.transform(
        #
        # plt.scatter(X_pca_array[:, 0], X_pca_array[:, 1], c=labels_cluster_gmm)

        # I think bars is the BIC score?

        return labels_cluster_gmm, bars


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

#from clustering.gaussian_mixture_modeling.GMM import n_components_range, bic, spl, bars, cv_types
#from main_run.clustering_run import plot_label



class MainMethods:
    '''
        Load data and scale non-binary variables
    '''
    def load_scaled_data(file_path):
        # read in file
        X= pd.read_csv(file_path).to_numpy()
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


    '''
        Determine distribution of candidates /  percent of candidates per cluster
    '''
    def assess_candidate_points(self, X, y, cluster_labels, model_name):
        candidate_count = {}
        file_path = '../model_statistics/' + model_name +'_stats.txt'

        with open(file_path, 'w') as f:
            for i in range(max(cluster_labels) + 1):
                points_in_cluster_label = y[np.where(cluster_labels == i)]
                (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
                frequencies = np.asarray((unique, counts)).T
                cluster_name = "\n\nCLUSTER " + str(i) + '\n'
                f.write(cluster_name)
                f.write(str(frequencies) +'\n')
                candidate_count[i] = frequencies

                # Assess cluster makeups
                cluster = X[np.where(cluster_labels == i)]
                df = pd.DataFrame(data=cluster, columns=["koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad"])
                # Averages
                f.write("AVERAGES\n koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}\n koi_prad: {}\n".format(
                    df["koi_fpflag_co"].mean(),
                    df["koi_fpflag_nt"].mean(),
                    df["koi_fpflag_ss"].mean(),
                    df["koi_fpflag_ec"].mean(),
                    df["koi_prad"].mean()))
                # Modes for Binary
                f.write("MODES\n koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}".format(
                    df["koi_fpflag_co"].mode(),
                    df["koi_fpflag_nt"].mode(),
                    df["koi_fpflag_ss"].mode(),
                    df["koi_fpflag_ec"].mode()))

                f.write("COUNT ZEROS\n koi_fpflag_co: {}\n koi_fpflag_nt: {}\n koi_fpflag_ss: {}\n koi_fpflag_ec: {}".format(
                    df["koi_fpflag_co"].value_counts(),
                    df["koi_fpflag_nt"].value_counts(),
                    df["koi_fpflag_ss"].value_counts(),
                    df["koi_fpflag_ec"].value_counts()))

        return candidate_count

    '''
        Plot first two PCA components
        CANDIDATES: Star points
        FALSE POSITIVES: dotted points
    '''
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

    '''
        Build Dendrogram for Hierarchical
    '''
    def build_dendrogram(self, X):
        sch.dendrogram(sch.linkage(X, method='ward'))
        plt.title('Dendrogram')
        plt.xlabel('Data Point')
        plt.ylabel('Euclidean Distances')
        plt.show()

    '''
        Plot PCA components (or dendrogram for hierarchical)
        Check distribution of CANDIDATE points in clusters
    '''
    def model_evaluation(self, X, y, cluster_labels, model_name, plot_flag):
        # Plot
        if (plot_flag):
            if (model_name == "hierarchical"):
                self.build_dendrogram(X)
            else:
                self.plot_pca(X, y, cluster_labels)
        # Assess candidate distribution
        self.assess_candidate_points(X, y, cluster_labels, model_name)


class ClusterModels:
    #dbscan model scaled data: X
    def dbscan_model(X):
        ep = 0.5
        min_samples = 6
        # Compute DBSCAN
        db = DBSCAN(eps=ep, min_samples=min_samples).fit(X)
        cluster_labels = db.labels_
        sc = metrics.silhouette_score(X, cluster_labels)
        print("Silhouette Coefficient: %0.3f" % sc)
        return cluster_labels, sc

    def kmeans_model(X, plot_label):
        # set parameters
        kmeans_kwargs = {
            'init': "random",
            'n_clusters': 15,
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42,
        }

        '''
        A list sse holds the SSE values for each k for elbow curve
        '''
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
        '''find elbow in elbow curve'''
        kl = KneeLocator(range(1, kmeans_kwargs['n_clusters']+1), sse, curve="convex", direction="decreasing")
        knee = kl.elbow
        #print("knee ", knee)


        ''' 
        second test is for the silhouette coefficients for each k
        '''
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

        print("Silhouette Coefficient: %0.3f" % max(silhouette_coefficients))
        kmeans_kwargs  = {
            'init': "random",
            'n_clusters': knee,
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42,
        }
        kmeans = KMeans(knee, **kmeans_kwargs)
        k_labels = kmeans.fit_predict(X)


        return k_labels, max(silhouette_coefficients)

    # Create clustering, check silhouette scores
    def hierarchical(X, plot_label):
        flag = True
        sils = []
        cluster = 6
        while (flag):
            # Compute Hierarchical Clustering
            hc = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='ward')
            hier_labels = hc.fit_predict(X)
            sils.append(metrics.silhouette_score(X, hier_labels))
            # print("Silhouette Coefficient: %0.3f"
            #       % metrics.silhouette_score(X, hier_labels))
            cluster += 1
            if (cluster > 25):
                break

        # Plot Silhouette Coefficients
        clusters = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        #sil_clusters = plt.plot(clusters, sils)
        if plot_label:
            plt.title('Silhouette Scores')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Score')
            plt.show()
        else:
            pass

        #run model based on highest sil score
        cluster = 6 + 1 + sils.index(max(sils))
        hc = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='ward')
        hier_labels = hc.fit_predict(X)

        #print(cluster, min(hier_labels))
        return hier_labels, max(sils)

    def gmm_model(X, plot_label):
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
        #print(len(bic))

        kl = KneeLocator(range(1, len(bic)+1), bic, curve="convex", direction="decreasing")
        knee = kl.elbow
        print('knee', knee)

        minVal = min(bic[:]);
        maxVal = max(bic[:]);
        bic_norm = (bic- minVal) / (maxVal - minVal)
        #bic_norm = [float(i)/max(bic) for i in bic]
        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])

        # Plot the BIC scores for each covariance type (star the lowest score)
        if plot_label:
            clf = low_gmm
            bars = []
            plt.figure(figsize=(8, 6))
            spl = plt.subplot(2, 1, 1)
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + .2 * (i - 2)
                bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                    (i + 1) * len(n_components_range)],
                    width=.2, color=color))

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
        #final_gmm = GaussianMixture(n_components=2)
        final_gmm = GaussianMixture(n_components=knee)
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
        norm_data = 1 - bic_norm[knee]
        print(bic_norm[knee-1])

        return labels_cluster_gmm, bic_norm[knee-1]





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn import metrics

from clustering_main import MainMethods
from clustering_main import ClusterModels


if __name__ == "__main__":

    # true or false plot figures or not (aside from the specific plot function)
    plot_label = False
    top5_filename = r"../data/exoplanet_cleanedrf_top_5.csv"
    top10_filename = r"../data/exoplanet_cleanedrf_top_10.csv"

    # Initialize other classes
    mm = MainMethods()
    cm = ClusterModels()

    #load the data
    scaled_top_10, top_10_labels  = mm.load_scaled_data(file_path=top10_filename)
    scaled_top_5, top_5_labels = mm.load_scaled_data(file_path=top5_filename)

    # Split data into train and test: ASSUMING TOP 5 FEATURES
    X_train, X_test, y_train, y_test = train_test_split(scaled_top_5, top_5_labels, test_size=0.2, random_state=42)

    '''
        DBSCAN
    '''
    dbscan_labels, dbscan_sc = cm.dbscan_model(X_train)
    # MODEL EVALUATION
    mm.model_evaluation(X_train, y_train, dbscan_labels, 'dbscan', plot_flag=False)
    # WRITE DBSCAN DATA
    data_w_labels = np.concatenate((X_train, dbscan_labels.reshape(dbscan_labels.shape[0], 1)), axis=1)
    data_w_labels = np.concatenate((y_train.reshape(y_train.shape[0], 1), data_w_labels), axis=1)
    clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
    clustered_df.to_csv("../data/dbscan_clustered_top_5.csv", index=False)

    '''
        KMEANS
    '''
    k_labels, scs = cm.kmeans_model(X_train, plot_label)
    # MODEL EVALUATION
    mm.model_evaluation(X_train, y_train, k_labels, 'kmeans', plot_flag=False)
    # WRITE KMEANS DATA
    data_w_labels = np.concatenate((X_train, k_labels.reshape(k_labels.shape[0], 1)), axis=1)
    data_w_labels = np.concatenate((y_train.reshape(y_train.shape[0], 1), data_w_labels), axis=1)
    clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
    clustered_df.to_csv("../data/kmeans_clustered_top_5.csv", index=False)

    '''
        HIERARCHICAL
    '''
    hier_labels, scs = cm.hierarchical(X_train)
    # MODEL EVALUATION
    mm.model_evaluation(X_train, y_train, hier_labels, 'hierarchical', plot_flag=False)
    # WRITE HIERARCHICAL DATA
    data_w_labels = np.concatenate((X_train, hier_labels.reshape(hier_labels.shape[0], 1)), axis=1)
    data_w_labels = np.concatenate((y_train.reshape(y_train.shape[0], 1), data_w_labels), axis=1)
    clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
    clustered_df.to_csv("../data/hierarchical_clustered_top_5.csv", index=False)

    '''
        GMM
    '''
    gmm_labels, bars = cm.gmm_model(X_train, plot_label)
    # MODEL EVALUATION
    mm.model_evaluation(X_train, y_train, gmm_labels, 'gmm', plot_flag=False)
    # WRITE HIERARCHICAL DATA
    data_w_labels = np.concatenate((X_train, gmm_labels.reshape(gmm_labels.shape[0], 1)), axis=1)
    data_w_labels = np.concatenate((y_train.reshape(y_train.shape[0], 1), data_w_labels), axis=1)
    clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
    clustered_df.to_csv("../data/gmm_clustered_top_5.csv", index=False)


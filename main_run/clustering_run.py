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
from main_run.clustering_main import MainMethods as mm

#set parameters
#true or false plot figures or not (aside from the specific plot function)
plot_label = True
top5_filename = r"../../data/exoplanet_cleanedrf_top_5.csv"
top10_filename = r"../../data/exoplanet_cleanedrf_top_10.csv"

#load the data
scaled_top_10, top_10_labels  = mm.load_scaled_data(top10_filename)
scaled_top_5, top_5_labels = mm.load_scaled_data(top5_filename)


#DBSCAN

top_10, top_5, labels = load_data()

dbscan_labels = dbscan(top_5, labels)
data_w_labels = np.concatenate((top_5, dbscan_labels.reshape(dbscan_labels.shape[0], 1)), axis=1)
data_w_labels = np.concatenate((labels.reshape(labels.shape[0], 1), data_w_labels), axis=1)

clustered_df = pd.DataFrame(data=data_w_labels, columns=["koi_pdisposition" ,"koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"])
clustered_df.to_csv("../../data/exoplanet_clustered_top_5.csv", index=False)


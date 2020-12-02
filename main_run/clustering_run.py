import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#sklearn imports
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn import metrics
from clustering_main import MainMethods
from clustering_main import ClusterModels


'''
set parameters
plot_label: true or false plot figures or not (aside from the specific plot function)
top_n_sc: number of top sc scores in order to choose which results to use downstream
'''
plot_label = False
top5_filename = r"../data/exoplanet_cleanedrf_top_5.csv"
top10_filename = r"../data/exoplanet_cleanedrf_top_10.csv"
top_n_sc = 3

mm = MainMethods()
cm = ClusterModels()
'''
load the data
'''
scaled_top_10, top_10_labels  = mm.load_scaled_data(file_path=top10_filename)
scaled_top_5, top_5_labels = mm.load_scaled_data(file_path=top5_filename)

'''
Run models for labels and sil coefficients
'''
'''DBSCAN'''
dbscan_labels_5, dbscan_sc_5 = cm.dbscan_model(scaled_top_5)
mm.plot_pca(scaled_top_5, top_5_labels, dbscan_labels_5, "DBSCAN ")
dbscan_labels_10, dbscan_sc_10 = cm.dbscan_model(scaled_top_10)
mm.plot_pca(scaled_top_10, top_10_labels, dbscan_labels_10, "DBSCAN ")
'''KMeans'''
kmeans_labels_5, kmeans_sc_5 = cm.kmeans_model(scaled_top_5, plot_label)
#mm.plot_pca(scaled_top_5, top_5_labels, kmeans_labels_5, "K-Means ")
kmeans_labels_10, kmeans_sc_10 = cm.kmeans_model(scaled_top_10, plot_label)
#mm.plot_pca(scaled_top_10, top_10_labels, kmeans_labels_10, "K-Means ")
'''Hierarchical'''
hier_labels_5, hier_sc_5 = cm.hierarchical(scaled_top_5,plot_label)
#mm.plot_pca(scaled_top_5, top_5_labels, hier_labels_5, "Hierarchical ")
hier_labels_10, hier_sc_10 = cm.hierarchical(scaled_top_10,plot_label)
#mm.plot_pca(scaled_top_10, top_10_labels, hier_labels_10, "Hierarchical ")
'''GMM'''
gmm_labels_5, gmm_bic_5 = cm.gmm_model(scaled_top_5,plot_label)
#mm.plot_pca(scaled_top_5, top_5_labels, gmm_labels_5, "Gaussian Mixture Modeling ")
gmm_labels_10, gmm_bic_10 = cm.gmm_model(scaled_top_10,plot_label)
#mm.plot_pca(scaled_top_10, top_10_labels, gmm_labels_10, "Gaussian Mixture Modeling ")

'''
dict to find best labels
'''
model_sc = {
    'dbscan_labels_5': dbscan_sc_5,
    'dbscan_labels_10': dbscan_sc_10,
    'kmeans_labels_5': kmeans_sc_5,
    'kmeans_labels_10': kmeans_sc_5,
    'hier_labels_5': hier_sc_5,
    'hier_labels_10': hier_sc_5,
    'gmm_labels_5': gmm_bic_5,
    'gmm_labels_10': gmm_bic_10
}

model_labels = {
    'dbscan_labels_5': dbscan_labels_5,
    'dbscan_labels_10': dbscan_labels_10,
    'kmeans_labels_5': kmeans_labels_5,
    'kmeans_labels_10': kmeans_labels_10,
    'hier_labels_5': hier_labels_5,
    'hier_labels_10': hier_labels_10,
    'gmm_labels_5': gmm_labels_5,
    'gmm_labels_10': gmm_bic_10
}
k = Counter(model_sc)
high = k.most_common(top_n_sc)

'''
Save results to file
'''

clustered_df_5 = pd.DataFrame(data = scaled_top_5, columns=["koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad"])
clustered_df_10 = pd.DataFrame(data = scaled_top_10, columns=["koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad","koi_depth","koi_teq","koi_impact","koi_insol","koi_model_snr"])
clustered_df_5["koi_pdisposition"] = top_5_labels
clustered_df_10["koi_pdisposition"] = top_10_labels

for i in high:
    print(i, type(i))
    name = i[0]
    clustered_df_5[name] = model_labels[name]
    clustered_df_10[name] = model_labels[name]

clustered_df_5.to_csv("../data/clustered_top_5_result.csv", index=False)
clustered_df_10.to_csv("../data/clustered_top_10_result.csv", index=False)



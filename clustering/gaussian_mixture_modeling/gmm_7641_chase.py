# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:00:19 2020

@author: Chase
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


from numpy import genfromtxt
exo_data = genfromtxt('exoplanet_cleanedrf_top_10.csv', delimiter=',')
#true_labels = genfromtxt('exoplanet_cleanedrf_w_score.csv',delimiter=',')

def gmm(data, components):
        # Normalize data for plotting and processing
        binary_vars = data[:, :4]
        scaled_numeric = StandardScaler().fit_transform(data[: , 4:])
        data = np.concatenate((binary_vars, scaled_numeric), axis = 1)
        # Fit a Gaussian mixture with EM using five components
        gmm = GaussianMixture(components, covariance_type='full').fit(data)
        labels = gmm.predict(data)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        plt.scatter(pca_data[:,0], pca_data[:, 1], c=labels, s=40, cmap='viridis')
        plt.title('PCA #1 & #2 Scatter Plot - GMM - 5 Features')
        plt.show()
        
        probs = gmm.predict_proba(data)
        #probs_binary = np.round(gmm.predict_proba(data),0)
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        return probs, aic, bic

def plot_gmm(exo_data):
    # Loop through GMM 1->10 times to get ideal number of clusters
    aic_vec = []
    bic_vec = []
    x_vec = []
    for i in range(10):
        probs, cur_aic, cur_bic = gmm(exo_data,i+1)
        aic_vec.append(cur_aic)
        bic_vec.append(cur_bic)
        x_vec.append(i)
    
    # Plot the AIC/BIC scores from each of the models with N # of clusters
    plt.plot(x_vec, aic_vec, 'r',label='AIC')
    plt.plot(x_vec, bic_vec, 'b',label='BIC')
    plt.legend()
    plt.title('AIC/BIC - 10 Feature Selection')
    plt.xlabel('# of Clusters')
    plt.ylabel('AIC/BIC Score')
    plt.show()

plot_gmm(exo_data)



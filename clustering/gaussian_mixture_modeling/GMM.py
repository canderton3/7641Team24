import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

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

#Determining number of components using BIC
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(scaled_no_score)
        bic.append(gmm.bic(scaled_no_score))
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
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

#Select 2 as the number of components and build final GMM model
final_gmm = GaussianMixture(n_components = 2)
final_gmm.fit(scaled_no_score)
labels_cluster_gmm = gmm.predict(scaled_no_score)
labels_cluster_gmm

#Build PCA for plotting
ndimensions = 2

pca = PCA(n_components=ndimensions)
pca.fit(scaled_no_score)
X_pca_array = pca.transform(scaled_no_score)
    
plt.scatter(X_pca_array[:,0], X_pca_array[:,1], c=labels_cluster_gmm)

# Find percent candidate / false positive in each cluster
for i in range (max(labels_cluster_gmm) + 1):
    points_in_cluster = scaled_no_score[np.where(labels_cluster_gmm == i)]
    points_in_cluster_label = labels[np.where(labels_cluster_gmm == i)]
    (unique, counts) = np.unique(points_in_cluster_label, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print("CLUSTER ", i)
    print(frequencies)

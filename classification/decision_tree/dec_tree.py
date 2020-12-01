import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics

def load_data():
    clustered_5 = pd.read_csv(r"../../data/exoplanet_cleanedrf_top_5.csv").to_numpy()
    # Get Labels
    labels = clustered_5[:, 0]
    # Delete Labels from data
    clustered_5 = np.delete(clustered_5, 0, axis=1)
    return clustered_5, labels

def dec_tree(X, y):
    X_df = X
    binary_vars = X[:, :4]

    # Scale non-binary variables
    scaled_numeric = StandardScaler().fit_transform(X[: , 4:])
    # Recombine
    X = np.concatenate((binary_vars, scaled_numeric), axis = 1)

    tree_params = {'criterion':['gini','entropy']}

    clf = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=10, n_jobs=-1)
    clf.fit(X, y)
    print(clf.cv_results_)

    '''fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                   feature_names=["koi_fpflag_co","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_ec","koi_prad", "cluster"],  
                   class_names=["koi_pdisposition"],
                   filled=True)'''

def test_score(clf):
    pass

if __name__ == "__main__":
    X, y = load_data()
    dec_tree(X, y)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics

def load_data():
    clustered_5 = pd.read_csv(r"../../data/dbscan_clustered_top_5.csv").to_numpy()
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

    #clf = DecisionTreeClassifier(random_state=0)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, y)
    print(clf.cv_results_)

    conf_matrix = metrics.plot_confusion_matrix(clf, X, y,
                                 cmap=plt.cm.Blues)
    plt.show()

def test_score(clf):
    pass

if __name__ == "__main__":
    X, y = load_data()
    dec_tree(X, y)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

def load_scaled_data(file_path):
    # read in file
    X = pd.read_csv(file_path).to_numpy()
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


if __name__ == "__main__":
    top5_filename = r"../data/dbscan_clustered_top_5.csv"
    top10_filename = r"../data/exoplanet_cleanedrf_top_10.csv"

    '''
        load the data
    '''
    '''scaled_top_10, top_10_labels  = load_scaled_data(top10_filename)'''
    scaled_top_5, top_5_labels = load_scaled_data(top5_filename)

    '''X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(scaled_top_10, top_10_labels, test_size=0.3, random_state=42)'''
    X_train, X_test, y_train, y_test = train_test_split(scaled_top_5, top_5_labels, test_size=0.25, random_state=50)

    '''
        DECISION TREES
    '''
    '''clf_dc_10 = DecisionTreeClassifier(random_state=42)
    clf_dc_10.fit(X_train_10, y_train_10)'''

    dt = DecisionTreeClassifier(criterion='entropy', random_state=12)
    dt.fit(X_train, y_train)

    dt_preds = dt.predict(X_test)

    print(classification_report(y_test, dt_preds))
    
    conf_matrix_dc = metrics.plot_confusion_matrix(dt, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_dc.ax_.set_title("Decision Tree Confusion Matrix")

    '''
        LOGISITIC REGRESSION
    '''
    '''clf_lr_10 = LogisticRegression(solver='liblinear', random_state=0)
    clf_lr_10.fit(X_train_10, y_train_10)'''

    lr = LogisticRegression(solver='liblinear', random_state=10)
    lr.fit(X_train, y_train)

    lr_preds = lr.predict(X_test)

    print(classification_report(y_test, lr_preds))

    conf_matrix_lr = metrics.plot_confusion_matrix(lr, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_lr.ax_.set_title("Logistic Regression Confusion Matrix")
    '''
        NAIVE BAYES
    '''
    gnb = GaussianNB()
    
    nb_preds = gnb.fit(X_train, y_train).predict(X_test)

    print(classification_report(y_test, nb_preds))

    conf_matrix_nb = metrics.plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_nb.ax_.set_title("Naive Bayes Confusion Matrix")

    plt.show()

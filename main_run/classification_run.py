import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
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
    X_train, X_test, y_train, y_test = train_test_split(scaled_top_5, top_5_labels, test_size=0.25, random_state=100)

    '''
        DECISION TREES
    '''
    '''clf_dc_10 = DecisionTreeClassifier(random_state=42)
    clf_dc_10.fit(X_train_10, y_train_10)'''

    '''#dt = DecisionTreeClassifier(criterion='entropy', random_state=12)
    dt = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=12)
    dt.fit(X_train, y_train)

    dt_preds = dt.predict(X_test)

    print(classification_report(y_test, dt_preds))
    
    conf_matrix_dc = metrics.plot_confusion_matrix(dt, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_dc.ax_.set_title("Gradient Boosted Tree Confusion Matrix")'''

    # encode string class values as integers
    label_encoded_y_test = LabelEncoder().fit_transform(y_test)
    label_encoded_y_train = LabelEncoder().fit_transform(y_train)
    '''
    # Grid Search for best parameters
    model = GradientBoostingClassifier()
    n_estimators = range(50, 400, 50)
    learning_rate = [0.001, 0.01, 0.1, 0.2]
    gbc_params = dict(n_estimators=n_estimators, learning_rate=learning_rate)

    kfold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    grid_search = GridSearchCV(model, gbc_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold_cv)
    grid_result = grid_search.fit(X_train, label_encoded_y_train)
    # Find best combo of parameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))'''

    gcb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100)
    gcb.fit(X_train, y_train)
    conf_matrix_xgb = metrics.plot_confusion_matrix(gcb, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_xgb.ax_.set_title("Gradient Boosted Classifier Confusion Matrix")

    '''
        XGBoost
    '''
    '''# encode string class values as integers
    label_encoded_y_test = LabelEncoder().fit_transform(y_test)
    label_encoded_y_train = LabelEncoder().fit_transform(y_train)
    # grid search
    model = XGBClassifier()
    n_estimators = range(50, 400, 50)
    learning_rate = [0.001, 0.01, 0.1, 0.2]
    xgb_params = dict(n_estimators=n_estimators, learning_rate=learning_rate)

    kfold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    grid_search = GridSearchCV(model, xgb_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold_cv)
    grid_result = grid_search.fit(X_train, label_encoded_y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))'''

    xgb = XGBClassifier(learning_rate=0.01, n_estimators=100)
    xgb.fit(X_train, y_train)
    conf_matrix_xgb = metrics.plot_confusion_matrix(xgb, X_test, y_test, cmap=plt.cm.Blues)
    conf_matrix_xgb.ax_.set_title("XGBoost Confusion Matrix")

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

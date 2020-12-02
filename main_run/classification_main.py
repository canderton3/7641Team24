import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
#sklearn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics


class MainMethods:
    def load_split_data(self, X):
        pass

    def evaluate_model(self, clf, test_x, test_y):
        pass

class ClassificationModels:
    def decision_tree(self, X, y):
        clf = DecisionTreeClassifier(random_state=412)
        clf.fit(X, y)
        print(clf.cv_results_)
        return clf

    def log_regression(self, X, y):
        # create model
        model = LogisticRegression(solver='liblinear', random_state=0).fit(X, y)
        return model
        # predict probability of our data
        predictions = model.predict_proba(X)
        binary_preds_str = model.predict(X)
        # convert binary predictions from strings to numeric data
        binary_preds = np.empty(len(binary_preds_str))
        for i in range(len(binary_preds_str)):
            if binary_preds_str[i] == 'CANDIDATE':
                binary_preds[i] = 1
            else:
                binary_preds[i] = 0

        
        # print model summary
        print(classification_report(y, model.predict(X)))
        score = round(model.score(X, y), 4) * 100
        print('The model accuracy is {}%'.format(score))

    def naive_bayes(self, X, y):
        pass
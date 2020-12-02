import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
 


def load_data():
    clustered_5_raw = 'https://raw.githubusercontent.com/canderton3/7641Team24/master/data/dbscan_clustered_top_5.csv'
    clustered_5 = pd.read_csv(clustered_5_raw).to_numpy()
    # Get Labels
    labels = clustered_5[:, 0]
    # Delete Labels from data
    clustered_5 = np.delete(clustered_5, 0, axis=1)
    return clustered_5, labels

def naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    gnb = GaussianNB()
    
    y_prediction_gnb = gnb.fit(X_train, y_train).predict(X_test)
    cnf_matrix_gnb = confusion_matrix(y_test, y_prediction_gnb)
    probas = gnb.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label = 'FALSE POSITIVE')
    
    print("The number of mislabeled planets using Gaussian Naive Bayes out of a total %d is %d"
          % (X_test.shape[0], (y_test != y_prediction_gnb).sum()))
    print('Accuracy for our test data is', accuracy_score(y_test, y_prediction_gnb))
    print(cnf_matrix_gnb)
    plt.plot([0,1],[0,1],'k--') #plot the diagonal line
    plt.plot(fpr, tpr, label='NB') #plot the ROC curve
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC Curve Naive Bayes')
    plt.show()
    
    
    
X, y = load_data()
naive_bayes(X, y)

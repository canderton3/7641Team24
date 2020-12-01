# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:52:26 2020

@author: Chase
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

def load_data():
    # returns top10/top5 rdf attributes as pandas, converting all text to numerical values (i.e. KOI)
    data = pd.read_csv('C:\\Users\\Chase\\Google Drive\\College\\Graduate\\Fall\\CS 7641 - Machine Learning\\Team Project\\Classification\\data\\exoplanet_cleanedrf_top_5.csv')
    data = data.to_numpy()
    print(type(data))
    
    labels = data[:,0]
    
    clustered = data[:,1:6]
    
    return labels, clustered, data

def logit(data, labels, plot=True):
    # seperate binary and numeric data for scaling
    binary = data[:,0:4]
    numeric = data[:,-1].reshape(-1,1)
    # scale numeric data
    scaled_numeric = StandardScaler().fit_transform(numeric)
    # recombine dataset
    dataset = np.concatenate((binary, scaled_numeric),axis=1)
    # seperate training and testing data
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.30, random_state=42)
    # create model
    model = LogisticRegression(solver='liblinear', random_state=0).fit(train_data, train_labels)
    # print model classes (candidate/non-candidate)
    model.classes_
    # predict probability of our data
    predictions = model.predict_proba(test_data)
    binary_preds_str = model.predict(test_data)
    # convert binary predictions from strings to numeric data
    binary_preds = np.empty(len(binary_preds_str))
    for i in range(len(binary_preds_str)):
        if binary_preds_str[i] == 'CANDIDATE':
            binary_preds[i] = 1
        else:
            binary_preds[i] = 0
    # print model summary
    print(classification_report(test_labels, model.predict(test_data)))
    score = round(model.score(test_data, test_labels),4)*100
    print('The model accuracy is {}%'.format(score))
    # check for plot boolean from input
    if plot:
        # generate confusion matrix plot
        cm = confusion_matrix(test_labels, model.predict(test_data))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.title('Confusion Matrix - Testing Data')
        plt.show()
        # generate roc curve
        test_labels_int = np.empty(len(test_labels))
        for i in range(len(test_labels)):
            if test_labels[i] == 'CANDIDATE':
                test_labels_int[i] = 1
            else:
                test_labels_int[i] = 0
        fpr, tpr, thresholds = metrics.roc_curve(test_labels_int, predictions[:,0])
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return score
    
labels, clustered, data = load_data()
score = logit(clustered, labels)
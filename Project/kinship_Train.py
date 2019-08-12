# Kinship Project ML training
# @ Michael

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mtmg
from scipy import signal
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import random
import os
from pathlib import Path
from glob import glob
import sys
import math
import scipy.stats as stats
import scipy.io as spo  # for loading matlab file
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

###############################################################################
# load the dataset and split the sample
###############################################################################

os.chdir('/Users/Michael/Documents/MachineLearning/Project')

# Load the dataset
data_frame = pd.read_csv('data_mat.csv', header=None)
data_frame.shape
data_mat = np.asmatrix(data_frame)

# test the data
trainX = data_frame.iloc[:, 0:200]
trainY = data_frame.iloc[:, -1]

# split the sample
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainX,
                                                    trainY,
                                                    test_size=0.3,
                                                    random_state=42)

###############################################################################
# LDA QDA, logistic, Bayes, KNN, SVM, Neutral network
###############################################################################

# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(X_train, y_train)

lda_fit_pred = lda_fit.predict(X_test)

print(classification_report(y_test, lda_fit_pred, digits=4))


# Compute ROC curve and ROC area for each class
fpr, tpr,_=roc_curve(lda_fit_pred ,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')

# # compute auc
# roc_auc_score(lda_fit_pred ,y_test)

# Use QDA to classify
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda_fit = qda.fit(X_train, y_train)

qda_fit_pred = qda_fit.predict(X_test)

print(classification_report(y_test, qda_fit_pred, digits=4))

fpr, tpr,_=roc_curve(qda_fit_pred ,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')

# # compute auc
# roc_auc_score(qda_fit_pred ,y_test)


# Use NGB to classify
from sklearn.naive_bayes import GaussianNB

NG_model = GaussianNB()
NG_model.fit(X_train, y_train)
NG_prediction = NG_model.predict(X_test)
print(classification_report(y_test, NG_prediction, digits=4))

fpr, tpr,_=roc_curve(NG_prediction,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=1, label='ROC curve')


# Logistic regression

from sklearn.linear_model import LogisticRegression

lgs = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial',
                         max_iter=200).fit(X_train, y_train)

lgs_predict = lgs.predict(X_test)

print(classification_report(y_test, lgs_predict, digits=4))



# KNN

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

knn_predict = neigh.predict(X_test)

print(classification_report(y_test, knn_predict, digits=4))

fpr, tpr,_=roc_curve(knn_predict,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=1, label='ROC curve')

# SVM

from sklearn import svm

clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_train, y_train)

clf_svm_pred = clf_svm.predict(X_test)

print(classification_report(y_test, clf_svm_pred, digits=4))

fpr, tpr,_=roc_curve(clf_svm_pred,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=1, label='ROC curve')


# SVM- scale data input
from sklearn import preprocessing

clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(preprocessing.scale(X_train), y_train)

clf_svm_pred = clf_svm.predict(preprocessing.scale(X_test))

print(classification_report(y_test, clf_svm_pred, digits=4))

fpr, tpr,_=roc_curve(clf_svm_pred,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=1, label='ROC curve')

# neutral network
from sklearn.neural_network import MLPClassifier

nnt = MLPClassifier(solver='lbfgs', alpha=1e-2,
                    hidden_layer_sizes=(12, 6), random_state=1)

nnt.fit(X_train, y_train)
nnt_pred = nnt.predict(X_test)


print(classification_report(y_test, nnt_pred, digits=4))

fpr, tpr,_=roc_curve(nnt_pred,y_test,drop_intermediate=False)
plt.plot(fpr, tpr, color='red',lw=1, label='ROC curve')

###############################################################################
# clasify based on distance - knn extenstion, one features
###############################################################################

datax_norm = np.zeros([data_mat.shape[0], 1])
for i in range(data_mat.shape[0]):
    datax_norm[i, :] = np.linalg.norm(data_mat[i, 0:200])

xnorm_train, xnorm_test, ynorm_train, ynrom_test = train_test_split(
    datax_norm[np.where(datax_norm <500)[0]],
    trainY[np.where(datax_norm <500)[0]], test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(xnorm_train, ynorm_train)

knn_predict = neigh.predict(xnorm_test)

print(classification_report(ynrom_test, knn_predict, digits=4))


# End of the code

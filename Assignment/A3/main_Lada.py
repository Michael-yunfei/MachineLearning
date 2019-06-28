# Machine Learning
# Assignment 3
# Wenxuan Zhang  01/945008
# Fei Wang 01/942870
# Lada Rudnitckaia 01/942458


# Generative classifiers for binary and multiclass classification with one
# and multiple features

## Part I: Load Data ======================================================
import numpy as np
import pandas as pd
import scipy.io as spo
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import statsmodels.api as sm
from mpl_toolkits import mplot3d
import time
import math
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#  Load the 'wine.csv' dataset
# please set your own working directory which inlcudes the dataset
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A3')


dataset = pd.read_csv('wine.csv', header = None)
columns_names = ['Class','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
             'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
             'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
dataset.columns = columns_names

# Create separate variables containing the class labels and all the
# available features.
classes = dataset[['Class']]
X_all = dataset.drop(columns='Class')

# and determine how many classes there are in the dataset.
num_classes = len(classes.Class.unique())
print('Number of classes:', num_classes)

# Create a variable containing the names of the features.
descrX = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
             'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
             'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print(classes)
print(X_all)

# Determine how many representatives of each class there are in the dataset
classes_freq = pd.DataFrame(classes['Class'].value_counts())
classes_freq = classes_freq.sort_index()

n1 = int(classes_freq.iloc[0,:])
n2 = int(classes_freq.iloc[1,:])
n3 = int(classes_freq.iloc[2,:])
print('Class 1:', n1)
print('Class 2:', n2)
print('Class 3:', n3)

n = len(classes)
print(n)

if n != n1 + n2 + n3:
    print('something wrong in the computation of class representatives')
else:
    print('Data succesfully loaded. \n')


## Part II: Binary Classification with One Feature ========================

# Select only classes 1 and 3 for this part and feature 'Proanthocyanins'.
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.
classes_to_choose = [1,3]
dataset_ = dataset[dataset.Class.isin(classes_to_choose)]

Y_1 = dataset_[['Class']]
featuresSet_1 = dataset_[['Proanthocyanins']]

Y_1 = Y_1.replace(1, 0)
Y_1 = Y_1.replace(3, 1)

classLabels_1 = [0,1]

features_Class_1 = dataset_[dataset_.Class == 1][['Proanthocyanins']]
features_Class_3 = dataset_[dataset_.Class == 3][['Proanthocyanins']]

###########################################################################
# Task 0: Plot the data by creating two <count> density-normalized histograms in
# two different colors of your choice; for that use the specific normalization
# and 'BinWidth' set to 0.3.
# Add the grid.
# Add descriptive legend and title.
###########################################################################
plt.style.use('seaborn-deep')

plt.hist(np.array(features_Class_1),density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_1))[0]-0.3,
                        max(np.array(features_Class_1))[0]+0.3,0.3))
plt.hist(np.array(features_Class_3),density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_3))[0]-0.3,
                        max(np.array(features_Class_3))[0]+0.3,0.3))

plt.legend(loc='upper right')
plt.title('Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


###########################################################################
# Task 1 : Construct LDA classifier. For that fill in the function fitLDA
def fitLDA(featuresSet, groupSet, classLabels):
    dt = np.concatenate((groupSet, featuresSet), axis=1)
    num_features = featuresSet.shape[1]
    num_classes = len(classLabels)
    mu_params = np.zeros((num_features, num_classes))
    pi_params = []
    num_per_class = []
    for c in range(num_classes):
        pi_params.append(dt[dt[:, 0] == c].shape[0]/groupSet.shape[0])
        num_per_class.append(dt[dt[:, 0] == c].shape[0])
    for f in range(num_features):
        for c in range(num_classes):
            dt_ = dt[dt[:, 0] == c]
            mu = dt_[:,f+1].mean()
            mu_params[f,c] = float(mu)

    sum_cov = np.zeros([num_features, num_features])
    for c in range(num_classes):
        mean_vector = mu_params[:,c].reshape(-1, num_features )
        n_features = dt[dt[:, 0] == c, 1:].shape[0]
        mean_matrix = np.repeat(mean_vector, n_features, axis=0)
        diff_matrix = dt[dt[:, 0] == c, 1:] - mean_matrix
        sigma_k = diff_matrix.T @ diff_matrix
        sum_cov += sigma_k
    cov_params = sum_cov/(featuresSet.shape[0]-num_classes)

    return(mu_params, cov_params, pi_params)


mu_par_1 = fitLDA(featuresSet_1, Y_1, classLabels_1)[0]
cov_par_1 = fitLDA(featuresSet_1, Y_1, classLabels_1)[1]
pi_par_1 = fitLDA(featuresSet_1, Y_1, classLabels_1)[2]

print('Task 1, mean values:\n', mu_par_1)
print('Task 1, covariance:\n', cov_par_1)
print('Task 1, prior probabilities:\n', pi_par_1)


# and classifyLDA.
def classifyLDA(featuresSet, mu_params, cov_params, pi_params, classLabels, computeCritValue):
    xCritValue = None
    classLabelsNumber = len(classLabels)
    numObs, numFeatures = np.shape(featuresSet)

    scores = np.zeros((numObs, classLabelsNumber))
    groupSet = np.zeros((numObs, 1))

    for c in range(classLabelsNumber):
        delta = (featuresSet @ np.linalg.inv(cov_params) @ mu_params[:,c] -
                 0.5 * mu_params[:,c].T @ np.linalg.inv(cov_params) @ mu_params[:,c] +
                 np.log(pi_params[c]))
        scores[:,c] = delta
    groupSet = np.argmax(scores, axis=1)

    if computeCritValue !=0 and numFeatures == 1:
        mu0 = mu_params[0,0]
        mu1 = mu_params[0,1]
        var = cov_params[0,0]
        pi0 = pi_params[0]
        pi1 = pi_params[1]
        xCritValue = var/(mu1-mu0) * ((mu1**2-mu0**2)/(2*var) + np.log((1-pi1)/pi1))

    return(groupSet, xCritValue)

y_pred_1 = classifyLDA(featuresSet_1, mu_par_1, cov_par_1, pi_par_1, classLabels_1,
                       computeCritValue = 1)[0]
xCritValue_1 = classifyLDA(featuresSet_1, mu_par_1, cov_par_1, pi_par_1, classLabels_1,
                           computeCritValue = 1)[1]

print('Task 1, classes predicted by Proanthocyanins using LDA:\n', y_pred_1)
print('Task 1, xCritValue_1:\n', xCritValue_1)
##########################################################################

###########################################################################
# Task 2 : Compute the empirical value of the error using the 0-1 loss.
# For that add typeOfLoss '0-1' option to the function computeLoss from the
# previous assignment. Additionally, this function needs to output the Type I and Type II
# errors (false positive and false negative) which will be filled in only in the case of
# binary classification.
##########################################################################
def computeLoss(Y, Y_pred, typeofLoss):
    Y = np.array(Y)
    Y_pred = np.array(Y_pred).reshape(-1,1)
    classLabelsNumber = len(np.unique(Y))
    FP = 0
    FN = 0
    if typeofLoss =='0-1 Loss':
        accuracy = np.equal(Y_pred, Y)
        accuracy = accuracy.astype(int)
        accuracy = np.array(accuracy)
        L = (accuracy[accuracy[:,0]==0].shape[0])/Y.shape[0]
        if classLabelsNumber == 2:
            for i in range(Y.shape[0]):
                if Y[i,0]==0 and Y_pred[i,0]==1:
                    FP = FP+1
                    i=i+1
                elif Y[i,0]==1 and Y_pred[i,0]==0:
                    FN = FN+1
                    i=i+1
            TIError = FP / Y[Y[:,0]==0].shape[0]
            TIIError = FN / Y[Y[:,0]==1].shape[0]
        else:
            TIError = None
            TIIError = None
    else:
         print('Type of loss is unknown')
    return(L, TIError, TIIError)

loss01_1 = computeLoss(Y_1, y_pred_1, typeofLoss='0-1 Loss')
print('Task 2, 0-1 loss:', loss01_1[0])
print('Task 2, Type I error:', loss01_1[1])
print('Task 2, Type II error:', loss01_1[2])

###########################################################################
# Task 3 : Plot the resulting classification.
# Create two histograms in two different colors of your choice: for these,
# use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# Superimpose the two normal distributions and the Gaussian mixture distribution that you
# obtain with the parameters computed in the 'fitLDA' function.
# Add the grid.
# Add descriptive legend and title.
# Plot the decision boundary (critical value for the given threshold of interest,
# which is set by default to 1)
##########################################################################
dataset_3 = pd.concat([featuresSet_1, Y_1], axis=1)

features_Class_0 = dataset_3[dataset_3.Class == 0][['Proanthocyanins']]
features_Class_1 = dataset_3[dataset_3.Class == 1][['Proanthocyanins']]

mu_1 = mu_par_1[0,0]
var_1 = np.array(features_Class_0).std()
min_1 = np.array(features_Class_0).min()
max_1 = np.array(features_Class_0).max()
mu_3 = mu_par_1[0,1]
var_3 = np.array(features_Class_1).std()
min_3 = np.array(features_Class_1).min()
max_3 = np.array(features_Class_1).max()
mu, cov = stats.norm.fit(np.array(featuresSet_1))
min_ = np.array(featuresSet_1).min()
max_ = np.array(featuresSet_1).max()

plt.style.use('seaborn-deep')
plt.hist(np.array(features_Class_0),density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_0))[0]-0.3,
                        max(np.array(features_Class_0))[0]+0.3,0.3))
plt.hist(np.array(features_Class_1),density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_1))[0]-0.3,
                        max(np.array(features_Class_1))[0]+0.3,0.3))
plt.axvline(xCritValue_1, color='r', label='LDA critical X')
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 3, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


###########################################################################
# Task 4 : Construct QDA classifier. For that fill in the function fitQDA
def fitQDA(featuresSet, groupSet, classLabels):
    dt = np.concatenate((groupSet, featuresSet), axis=1)
    num_features = featuresSet.shape[1]
    num_classes = len(classLabels)
    mu_params = np.zeros((num_features, num_classes))
    pi_params = []
    cov_params = []
    for c in range(num_classes):
        featuresSet_c = dt[dt[:, 0] == c, 1:]
        pi_params.append(dt[dt[:, 0] == c].shape[0]/groupSet.shape[0])
        if num_features == 1:
            covar = np.array(np.cov(featuresSet_c.T)).reshape(1,1)
        else:
            covar = np.cov(featuresSet_c.T)
        cov_params.append(covar)
    for f in range(num_features):
        for c in range(num_classes):
            dt_ = dt[dt[:, 0] == c]
            mu = dt_[:,f+1].mean()
            mu_params[f,c] = float(mu)

    return(mu_params, cov_params, pi_params)

mu_par_4 = fitQDA(featuresSet_1, Y_1, classLabels_1)[0]
cov_par_4 = fitQDA(featuresSet_1, Y_1, classLabels_1)[1]
pi_par_4 = fitQDA(featuresSet_1, Y_1, classLabels_1)[2]

print('Task 4, mean values:\n', mu_par_4)
print('Task 4, covariance:\n', cov_par_4)
print('Task 4, prior probabilities:\n', pi_par_4)

# and classifyQDA.
def classifyQDA(featuresSet, mu_params, cov_params, pi_params, classLabels):
    classLabelsNumber = len(classLabels)
    numObs, numFeatures = np.shape(featuresSet)

    delta = np.zeros((numObs, classLabelsNumber))
    groupSet = np.zeros((numObs, 1))

    for c in range(classLabelsNumber):
        det_part = -0.5*np.log(np.linalg.det(cov_params[c]))
        pi_part = np.log(pi_params[c])
        mu = mu_params[:,c]
        for i in range(numObs):
            x = featuresSet[i,:]
            x_mu = x - mu.T
            x_mu_sig_part = -0.5*x_mu @ np.linalg.det(cov_params[c]) @ x_mu.T
            delta_i = det_part + x_mu_sig_part + pi_part
            delta[i,c] = delta_i

    groupSet = np.argmax(delta, axis=1)

    return(groupSet)

y_pred_4 = classifyQDA(featuresSet_1, mu_par_4, cov_par_4, pi_par_4, classLabels_1)

print('Task 4, classes predicted by Proanthocyanins using QDA:\n', y_pred_4)
##########################################################################

###########################################################################
# Task 5 : Compute the empirical value of the error using the 0-1 loss.
# using the function computeLoss together with the Type I and Type II errors
##########################################################################
loss01_5 = computeLoss(Y_1, y_pred_4, typeofLoss='0-1 Loss')
print('Task 5, 0-1 loss:', loss01_5[0])
print('Task 5, Type I error:', loss01_5[1])
print('Task 5, Type II error:', loss01_5[2])

###########################################################################
# Task 6 : Plot the resulting classification.
# Create two histograms in two different colors of your choice: for these,
# use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# Superimpose the two normal distributions and the mixed distribution that you
# obtain as a result from the 'fitLDA' function.
# Add the grid.
# Add descriptive legend and title.
##########################################################################
dataset_6 = pd.concat([featuresSet_1, Y_1], axis=1)

features_Class_0 = dataset_6[dataset_6.Class == 0][['Proanthocyanins']]
features_Class_1 = dataset_6[dataset_6.Class == 1][['Proanthocyanins']]

mu_1 = mu_par_4[0,0]
var_1 = np.array(features_Class_0).std()
min_1 = np.array(features_Class_0).min()
max_1 = np.array(features_Class_0).max()
mu_3 = mu_par_4[0,1]
var_3 = np.array(features_Class_1).std()
min_3 = np.array(features_Class_1).min()
max_3 = np.array(features_Class_1).max()
mu = np.array(featuresSet_1).mean()
cov = np.array(featuresSet_1).std()
min_ = np.array(featuresSet_1).min()
max_ = np.array(featuresSet_1).max()
mu_1, var_1 = stats.norm.fit(np.array(features_Class_0))
mu_3, var_3 = stats.norm.fit(np.array(features_Class_1))
mu, cov = stats.norm.fit(np.array(featuresSet_1))

plt.style.use('seaborn-deep')
plt.hist(np.array(features_Class_0),density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_0))[0]-0.3,
                        max(np.array(features_Class_0))[0]+0.3,0.3))
plt.hist(np.array(features_Class_1),density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(np.array(features_Class_1))[0]-0.3,
                        max(np.array(features_Class_1))[0]+0.3,0.3))
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 6, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


###########################################################################
# Task 7 : Construct Naive Bayes Gauss classifier. For that fill in the
# function fitNaiveBayesGauss
def fitNaiveBayesGauss(featuresSet, groupSet, classLabels):
    dt = np.concatenate((groupSet, featuresSet), axis=1)
    num_features = featuresSet.shape[1]
    num_classes = len(classLabels)
    mu_params = np.zeros((num_features, num_classes))
    pi_params = []
    cov_params = []
    for c in range(num_classes):
        featuresSet_c = dt[dt[:, 0] == c, 1:]
        pi_params.append(dt[dt[:, 0] == c].shape[0]/groupSet.shape[0])
        if num_features == 1:
            covar = np.array(np.cov(featuresSet_c.T)).reshape(1,1)
        else:
            covar = np.cov(featuresSet_c.T)
        for i in range(covar.shape[0]):
            for j in range(covar.shape[0]):
                if i != j:
                    covar[i,j] = 0
        cov_params.append(covar)
    for f in range(num_features):
        for c in range(num_classes):
            dt_ = dt[dt[:, 0] == c]
            mu = dt_[:,f+1].mean()
            mu_params[f,c] = float(mu)

    return(mu_params, cov_params, pi_params)

mu_par_7 = fitNaiveBayesGauss(featuresSet_1, Y_1, classLabels_1)[0]
cov_par_7 = fitNaiveBayesGauss(featuresSet_1, Y_1, classLabels_1)[1]
pi_par_7 = fitNaiveBayesGauss(featuresSet_1, Y_1, classLabels_1)[2]

print('Task 7, mean values:\n', mu_par_7)
print('Task 7, covariance:\n', cov_par_7)
print('Task 7, prior probabilities:\n', pi_par_7)

# and classifyNaiveBayesGauss.
def classifyNaiveBayesGauss(featuresSet, mu_params, cov_params, pi_params, classLabels):
    classLabelsNumber = len(classLabels)
    numObs, numFeatures = np.shape(featuresSet)

    scores = np.zeros((numObs, classLabelsNumber))
    groupSet = np.zeros((numObs, 1))

    for c in range(classLabelsNumber):
        delta = (featuresSet @ np.linalg.inv(cov_params[c]) @ mu_params[:,c] -
                 0.5 * mu_params[:,c].T @ np.linalg.inv(cov_params[c]) @ mu_params[:,c] +
                 np.log(pi_params[c]))
        scores[:,c] = delta
    groupSet = np.argmax(scores, axis=1)

    return(groupSet)

y_pred_7 = classifyNaiveBayesGauss(featuresSet_1, mu_par_7, cov_par_7, pi_par_7, classLabels_1)

print('Task 7, classes predicted by Proanthocyanins using Gaussian Naive Bayes:\n', y_pred_7)

#########################################################################

###########################################################################
# Task 8 : Compute the empirical value of the error using the 0-1 loss.
# using the function computeLoss together with the Type I and Type II
# errors
##########################################################################
loss01_8 = computeLoss(Y_1, y_pred_7, typeofLoss='0-1 Loss')
print('Task 8, 0-1 loss:', loss01_8[0])
print('Task 8, Type I error:', loss01_8[1])
print('Task 8, Type II error:', loss01_8[2])

###########################################################################
# Task 9 : Plot the resulting classification.
# Create two histograms in two different colors  of your choice: for these,
# use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# Superimpose the two normal distributions and the mixed distribution that you
# obtain as a result from the 'fitNaiveBayesGauss' function.
# Add the grid.
# Add descriptive legend and title.
##########################################################################



## Part III: Binary Classification with Two Features ======================
# Select only classes 1 and 3 for this part and features:
#   - 'Proanthocyanins'
#   - 'Alcalinity of ash'
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.
classes_to_choose = [1,3]
dataset_ = dataset[dataset.Class.isin(classes_to_choose)]

Y_10 = dataset_[['Class']]
featuresSet_10 = dataset_[['Proanthocyanins','Alcalinity of ash']]

Y_10 = Y_10.replace(1, 0)
Y_10 = Y_10.replace(3, 1)

classLabels_10 = [0,1]

###########################################################################
# Task 10 : Construct Naive Bayes Gauss classifier.
##########################################################################
mu_par_10 = fitNaiveBayesGauss(featuresSet_10, Y_10, classLabels_10)[0]
cov_par_10 = fitNaiveBayesGauss(featuresSet_10, Y_10, classLabels_10)[1]
pi_par_10 = fitNaiveBayesGauss(featuresSet_10, Y_10, classLabels_10)[2]

print('Task 10, mean values:\n', mu_par_10)
print('Task 10, covariance:\n', cov_par_10)
print('Task 10, prior probabilities:\n', pi_par_10)

y_pred_10 = classifyNaiveBayesGauss(featuresSet_10, mu_par_10, cov_par_10, pi_par_10, classLabels_10)

print('Task 10, classes predicted by Proanthocyanins and Alcalinity of ash using Gaussian Naive Bayes:\n', y_pred_10)

###########################################################################
# Compute the empirical value of the error using the 0-1 loss.
##########################################################################
loss01_10 = computeLoss(Y_10, y_pred_10, typeofLoss='0-1 Loss')
print('Task 10, 0-1 loss:', loss01_10[0])
print('Task 10, Type I error:', loss01_10[1])
print('Task 10, Type II error:', loss01_10[2])


###########################################################################
# Plot the resulting classification.
# Add the grid.
# Add descriptive legend and title.
# Mark misclassified observations.
##########################################################################




## Part IV: Binary Classification with Many Features ======================
# Select only classes 1 and 3 for this part and features:
#   - 'Alcohol'
#   - 'Flavanoids'
#   - 'Proanthocyanins'
#   - 'Color intensity'
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.
classes_to_choose = [1,3]
dataset_ = dataset[dataset.Class.isin(classes_to_choose)]

Y_11 = dataset_[['Class']]
featuresSet_11 = dataset_[['Alcohol','Flavanoids','Proanthocyanins','Color intensity']]

Y_11 = Y_11.replace(1, 0)
Y_11 = Y_11.replace(3, 1)

classLabels_11 = [0,1]

#########################################################################
# Task 11 : Construct LDA classifier.
##########################################################################
mu_par_11 = fitLDA(featuresSet_11, Y_11, classLabels_11)[0]
cov_par_11 = fitLDA(featuresSet_11, Y_11, classLabels_11)[1]
pi_par_11 = fitLDA(featuresSet_11, Y_11, classLabels_11)[2]

print('Task 11, mean values:\n', mu_par_11)
print('Task 11, covariance:\n', cov_par_11)
print('Task 11, prior probabilities:\n', pi_par_11)

y_pred_11 = classifyLDA(featuresSet_11, mu_par_11, cov_par_11, pi_par_11, classLabels_11,
                        computeCritValue = 1)[0]

print('Task 11, classes predicted by Alcohol, Flavanoids, Proanthocyanins and Color intensity using LDA:\n', y_pred_11)

# check with sklearn
clf_11 = LinearDiscriminantAnalysis().fit(np.array(featuresSet_11),
                                 np.array(Y_11).reshape(np.array(Y_11).shape[0],))
y_pred_clf_11 = np.array(clf_11.predict(featuresSet_11))

###########################################################################
# Compute the empirical value of the error using the 0-1 loss.
##########################################################################
loss01_11 = computeLoss(Y_11, y_pred_11, typeofLoss='0-1 Loss')
print('Task 11, 0-1 loss:', loss01_11[0])
print('Task 11, Type I error:', loss01_11[1])
print('Task 11, Type II error:', loss01_11[2])

loss01_11_clf = computeLoss(Y_11, y_pred_clf_11, typeofLoss='0-1 Loss')
print('Task 11, sklearn 0-1 loss:', loss01_11_clf[0])
print('Task 11, sklearn Type I error:', loss01_11_clf[1])
print('Task 11, sklearn Type II error:', loss01_11_clf[2])


## Part V: 3-Classes Classification with Many Features ====================
# Select only classes 1 and 3 for this part and features:
#   - 'Alcohol'
#   - 'Flavanoids'
#   - 'Proanthocyanins'
#   - 'Color intensity'
Y_12 = dataset[['Class']]
featuresSet_12 = dataset[['Alcohol','Flavanoids','Proanthocyanins','Color intensity']]

Y_12 = Y_12.replace(1, 0)
Y_12 = Y_12.replace(2, 1)
Y_12 = Y_12.replace(3, 2)

classLabels_12 = [0,1,2]

###########################################################################
# Task 12 : Construct QDA classifier for the following:
#   - 'Alcohol' ????????????
#   - 'Alcohol' + 'Proanthocyanins' ????????????
#   - All features listed above
mu_par_12 = fitQDA(featuresSet_12, Y_12, classLabels_12)[0]
cov_par_12 = fitQDA(featuresSet_12, Y_12, classLabels_12)[1]
pi_par_12 = fitQDA(featuresSet_12, Y_12, classLabels_12)[2]

print('Task 12, mean values:\n', mu_par_12)
print('Task 12, covariance:\n', cov_par_12)
print('Task 12, prior probabilities:\n', pi_par_12)

y_pred_12 = classifyQDA(featuresSet_12, mu_par_12, cov_par_12, pi_par_12, classLabels_12)

print('Task 12, classes predicted by Alcohol, Flavanoids, Proanthocyanins, Color intensity using QDA:\n', y_pred_4)
##########################################################################

###########################################################################
# Compute the empirical value of the errors using the 0-1 loss.
##########################################################################
loss01_12 = computeLoss(Y_12, y_pred_12, typeofLoss='0-1 Loss')
print('Task 12, 0-1 loss:', loss01_12[0])
print('Task 12, Type I error:', loss01_12[1])
print('Task 12, Type II error:', loss01_12[2])

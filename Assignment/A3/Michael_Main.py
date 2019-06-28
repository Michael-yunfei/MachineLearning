# Assignment 3 - Classification
# @ Coco, Lada, Michael

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import random
import os
import sys
import math
import scipy.stats as stats

# check working directory
os.getcwd()
# os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A3')
os.chdir('C:/Users/User/Desktop/ML/assignments/3')

###############################################################################
# Part I - Load the Dataset, select two types (classes) of wine.
###############################################################################
indexNames = ['WineClass', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium',
              'Total phenols', 'Flavanoids',
              'Nonflavanoid phenols',
              'Proanthocyanins', 'Color intensity',
              'Hue', 'OD280/OD315 of diluted wines',
              'Proline']
len(indexNames)

wine = pd.read_csv('wine.csv', names=indexNames)
wine.shape  # (178, 14)
wine.head()

# check whether dataset is balanced or not:
# If it is (perfectly) balanced, each class of Y should have the same
# number of observations. This is perfect for training our model
# If it is not perfectly balanced, we have to be careful: e.g.
# if Y=0 has 50 observations, Y=1 has 20 observations, Y=2 has 10 observations
# What should we do for this case?

uniqueY = np.unique(wine.WineClass)  # get unique values
len(uniqueY)  # 3
for i in range(len(uniqueY)):
    print(wine[wine.WineClass == uniqueY[i]].shape)

# (59, 14)
# (71, 14)
# (48, 14)
# The dataset is not prefectly balanced, but the differences of number of
# observations for three classes are not that big.

# Now we select only classes 1, 3 for this part and feature'Proanthocyanins'.
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.

wineSubset = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
wineSubset = wineSubset[['WineClass', 'Proanthocyanins']]
wineSubset['newclass'] = np.where(wineSubset['WineClass'] == 1, 0, 1)
wineSubset = wineSubset[['newclass', 'Proanthocyanins']]
wineSubset.shape
wineSubset.head()

features_Class_1 = np.array(wineSubset.query('newclass==0')['Proanthocyanins'])
features_Class_3 = np.array(wineSubset.query('newclass==1')['Proanthocyanins'])

###############################################################################
# Part II: Binary Classification with One Feature
###############################################################################

#  Task 0
#  Plot the data by creating two <count> density-normalized histograms
#  in two different colors of your choice; for that use the specific
#  normalization and 'BinWidth' set to 0.3. Add the grid.
#  Add descriptive legend and title.
plt.style.use('seaborn-deep')

plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))

plt.legend(loc='upper right')
plt.title('Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


# Task 1 :
# Construct LDA classifier. For that fill in the function fitLDA
# and classifyLDA. Both functions should be constructed in order to
# work with multiple classes and multiple feautures if needed. We start
# here however with only two-classes classification which admits the
# explicit critical decision boundary value.

# Function to fit LDA
def fitLDA(X, Y):
    '''
    X and Y are dataframe
    need to be transfomed into:
    Input: X - n by m matrix, containing all features
           Y - n by 1 matrix, contanning the information of class
    Causion: the orders of rows of X and Y are assumed to be corresponding to
             each other, which means inforamtion of any row (say row 10) of X
             and information of same row (row 10) of Y are from same
             observation
    Output: prior_probability: estimated probability: pi_k (dataframe)
            classmean: estiamted mean of different classes: mu_k (dataframe)
            sigma: estimated covariance matrix (matrix)
    '''
    n = X.shape[0]  # get the number of total observation
    columnsIndex = X.columns
    m = X.shape[1]  # get the number of covariate variables (features)
    X = np.asmatrix(X).reshape(n, m)
    Y = np.asarray(Y).reshape(-1, 1)  # Y has to be an array, not matrix
    yunique = np.unique(Y)
    prior_probability = {}  # initialize a distionary for prior probability
    mu_k = {}
    sigma = {}

    for i in yunique:
        rowindex = np.where(Y == i)[0]
        classlabel = 'class' + str(i)
        prior_probability[classlabel] = len(rowindex)/n
        filteredx = X[rowindex, :]
        mean_vector = np.mean(filteredx, axis=0)
        mu_k[classlabel] = mean_vector
        mean_maxtrix = np.repeat(mean_vector, filteredx.shape[0], axis=0)
        diff_matrix = filteredx - mean_maxtrix
        sigma_k = diff_matrix.T @ diff_matrix
        # calcluate within-class covariance
        sigma[classlabel] = sigma_k

    # tidy the output
    sigma_sum = np.zeros([m, m])
    for i in sigma:
        sigma_sum += sigma[i]
    sigma = sigma_sum/(n - len(yunique))  # estimate final sigma
    prior_probability = pd.DataFrame(list(prior_probability.values()),
                                     index=prior_probability.keys(),
                                     columns=['Prior Probability'])
    mean_dataframe = []
    for v in mu_k:
        mean_dataframe.extend(np.array(mu_k[v]))
    mu_k = pd.DataFrame(mean_dataframe, index=mu_k.keys(),
                        columns=columnsIndex)

    return(prior_probability, mu_k, sigma)


# Function to classify LDA
def classifyLDA(featureX, priorpro, mu, sigma, critical=False):
    '''
    This is the classification for multi-categories case
    and it only takes the binary case as a special one

    Input: 1)featureX: n by m dataframe, where n is sample size, m is number
           of covariatle variables

           2)mu: k by m dataframe, where k is the number of classes or
           categories m is the number of covariatle variables. mu is
           taken from fitLDA() function.

           3) priorpro: k by 1 dataframe, prior probabilty, it is taken from
           fitLDA() function.

           4) sigma: k by k covariance matrix, it is also taken from fitLDA()

           5) critical=Faulse, if it is true, then it should be the case that
              number of calsses = number of features. Otherwise, there is
              no solution for critical values.
              WARNING: in this function, the critical value calculation only
                       applies for the binar case with one feature
    Output:
           Classification results: n by 1 vector and newdataframe with
                                   extra column called 'LDAClassification'
           and k by 1 vector of critical values of X
    '''
    newX = pd.DataFrame.copy(featureX)
    classLabels = priorpro.index  # get class labes from dataframe
    featureLabels = featureX.columns
    meanLabels = mu.columns
    X = np.asmatrix(featureX)
    priorpro = np.asmatrix(priorpro)
    if all(featureLabels == meanLabels):
        delta = np.zeros([featureX.shape[0], 1])
        for v in range(len(classLabels)):
            Probabilty = np.array(priorpro[v, :]).reshape(-1, 1)
            # get prior probabilty for class k
            mean_vector = np.array(mu.iloc[v, :]).reshape(-1, 1)
            # get mean vector for class k
            deltaX = (X @ np.linalg.inv(sigma) @ mean_vector
                      - 1/2 * mean_vector.T @
                      np.linalg.inv(sigma) @ mean_vector
                      + math.log(Probabilty))
            delta = np.hstack([delta, np.asmatrix(deltaX).reshape(-1, 1)])

        delta = delta[:, 1:]
        classificationResults = np.argmax(delta, axis=1)
        # maximize the delta over k
        newX['LDAClassification'] = classificationResults.reshape(-1, 1)
    else:
        print('Pleasre make sure that featured X and mean vector\
              have the same covariate variables')
        sys.exist(0)

    if critical is True:
        if len(classLabels) < len(featureLabels):
            print('There is no solutions for critical values\
                  as dimension of classes is less than dimension\
                  of covariate variables')
            sys.exist(0)
        else:
            # calculate the critical values
            mean_i = np.array(mu.iloc[0, :]).reshape(-1, 1)
            mean_j = np.array(mu.iloc[1, :]).reshape(-1, 1)
            prob_i = np.array(priorpro[0, :]).reshape(-1, 1)
            prob_j = np.array(priorpro[1, :]).reshape(-1, 1)
            xcritical = sigma/(mean_j - mean_i)*(
                math.log(prob_i/prob_j) + (mean_j**2
                                         - mean_i**2)/(2*sigma))
            return(classificationResults, newX, xcritical)
    else:
        return(classificationResults, newX)


# Function to compute 0-1 loss and type I/II error for classification
def computeLoss(y, yhat, binary=False):
    '''
    Input: true classification y (dataframe n by 1)
           estimated classifcaiton yhat (dataframe n by 1)
           binary=False: if it is faulse, only zeroone loss is returned
                         if it is true, the class of y has to be binary,
                         then type I/II error table is returned
    Output:
          0-1 loss (dataframe) if binary=False
          type I/II error (dataframe) if binary=True
    '''
    n = y.shape[0]
    y = np.array(y).reshape(-1, 1)
    yhat = np.array(yhat).reshape(-1, 1)
    if binary is True:
        yunique = np.unique(y)
        y1Index = np.where(y == yunique[1])
        predict_positiveRatio = np.sum(yhat[y1Index] == yunique[1])/len(y1Index[0])
        type2error = 1 - predict_positiveRatio
        y0Index = np.where(y == yunique[0])
        predict_negativeRatio = np.sum(yhat[y0Index] == yunique[0])/len(y0Index[0])
        type1error = 1 - predict_negativeRatio
        typeError = pd.DataFrame(np.array([[predict_positiveRatio,
                                            type2error],
                                           [predict_negativeRatio,
                                            type1error]]),
                                 columns=['Predicted positive',
                                          'Predicted negative'],
                                 index=['class'+str(yunique[1]),
                                        'class'+str(yunique[0])])
        return(typeError)
    else:
        zeroOneloss = 1 - (sum(y == yhat)/n)
        return(zeroOneloss)


# test LDA with rull dataset
# prepare the dataset
trainX = pd.DataFrame(wineSubset.Proanthocyanins,
                      columns=['Proanthocyanins'])
trainY = pd.DataFrame(wineSubset.newclass,
                      columns=['newclass'])


binary_pro, binary_mean, binary_sigma = fitLDA(trainX, trainY)

preictedY, newX, critX = classifyLDA(trainX, binary_pro, binary_mean,
                                     binary_sigma, critical=True)
#
# % Task 2 : Compute the empirical value of the error using the 0-1 loss.
# % For that add typeOfLoss '0-1' option to the function computeLoss from the
# % previous assignment. Additionally, this function needs to output the Type I and Type II
# % errors (false positive and false negative) which will be filled in only in the case of
# % binary classification.

loss = computeLoss(trainY, preictedY)
print(loss)  # [0.1682243]
typeloss = computeLoss(trainY, preictedY, True)
print(typeloss)

# ###############################################################################
# % Task 3 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the Gaussian mixture distribution that you
# % obtain with the parameters computed in the 'fitLDA' function.
# % Add the grid.
# % Add descriptive legend and title.
# % Plot the decision boundary (critical value for the given threshold of interest,
# % which is set by default to 1)
# ###############################################################################
X = np.asmatrix(wineSubset.Proanthocyanins).reshape(-1, 1)
Y = np.asmatrix(wineSubset.newclass).reshape(-1, 1)
mu_1 = np.array(binary_mean)[0,0]
var_1 = np.array(features_Class_1).std()
min_1 = np.array(features_Class_1).min()
max_1 = np.array(features_Class_1).max()
mu_3 = np.array(binary_mean)[1,0]
var_3 = np.array(features_Class_3).std()
min_3 = np.array(features_Class_3).min()
max_3 = np.array(features_Class_3).max()
mu, cov = stats.norm.fit(np.array(X))
min_ = np.array(X).min()
max_ = np.array(X).max()

plt.style.use('seaborn-deep')
plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))
plt.axvline(critX, color='r', label='LDA critical X')
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 3, LDA, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


# ###############################################################################
# % Task 4 : Construct QDA classifier. For that fill in the function fitQDA
# % and classifyQDA. Both functions should be constructed in order to
# % work with multiple classes and multiple features if needed. We start
# % here however with only two-classes classification first.
# ###############################################################################
# Define QDA function
def fitQDA(X, Y):
    '''
    X and Y are dataframe
    need to be transfomed into:
    Input: X - n by m matrix, containing all features
           Y - n by 1 matrix, contanning the information of class
    Causion: the orders of rows of X and Y are assumed to be corresponding to
             each other, which means inforamtion of any row (say row 10) of X
             and information of same row (row 10) of Y are from same
             observation
    Output: prior_probability: estimated probability: pi_k (dataframe)
            classmean: estiamted mean of different classes: mu_k (dataframe)
            sigma: estimated covariance matrix (matrix)
    '''
    n = X.shape[0]  # get the number of total observation
    columnsIndex = X.columns
    m = X.shape[1]  # get the number of covariate variables (features)
    X = np.asmatrix(X).reshape(n, m)
    Y = np.asarray(Y).reshape(-1, 1)  # Y has to be an array, not matrix
    yunique = np.unique(Y)
    prior_probability = {}  # initialize a distionary for prior probability
    mu_k = {}
    sigma = {}

    for i in yunique:
        rowindex = np.where(Y == i)[0]
        classlabel = 'class' + str(i)
        prior_probability[classlabel] = len(rowindex)/n
        filteredx = X[rowindex, :]
        mean_vector = np.mean(filteredx, axis=0)
        mu_k[classlabel] = mean_vector
        mean_maxtrix = np.repeat(mean_vector, filteredx.shape[0], axis=0)
        diff_matrix = filteredx - mean_maxtrix
        sigma_k = (diff_matrix.T @ diff_matrix)/(n-1)
        # calcluate within-class covariance
        sigma[classlabel] = sigma_k

    # tidy the output
    prior_probability = pd.DataFrame(list(prior_probability.values()),
                                     index=prior_probability.keys(),
                                     columns=['Prior Probability'])
    mean_dataframe = []
    for v in mu_k:
        mean_dataframe.extend(np.array(mu_k[v]))
    mu_k = pd.DataFrame(mean_dataframe, index=mu_k.keys(),
                        columns=columnsIndex)

    return(prior_probability, mu_k, sigma)


# Define QDA calssification
def classifyQDA(featureX, priorpro, mu, sigma):
    '''
    This is the classification for multi-categories case
    and it only takes the binary case as a special one

    Input: 1)featureX: n by m dataframe, where n is sample size, m is number
           of covariatle variables

           2)mu: k by m dataframe, where k is the number of classes or
           categories m is the number of covariatle variables. mu is
           taken from fitLDA() function.

           3) priorpro: k by 1 dataframe, prior probabilty, it is taken from
           fitLDA() function.

           4) sigma: k by 1 dictionary, each erray is a m by m matrix
    Output:
           Classification results: n by 1 vector and newdataframe with
                                   extra column called 'LDAClassification'
           and k by 1 vector of critical values of X
    '''
    newX = pd.DataFrame.copy(featureX)
    classLabels = priorpro.index  # get class labes from dataframe
    featureLabels = featureX.columns
    meanLabels = mu.columns
    X = np.asmatrix(featureX)
    if all(featureLabels == meanLabels):
        delta = np.zeros([featureX.shape[0], 1])
        deltaX = np.zeros([featureX.shape[0], 1])
        for v in classLabels:
            probabilty = np.array(priorpro.loc[v, :])
            mean_vector = np.array(mu.loc[v, :]).reshape(-1, 1)
            sigma_k = sigma[v]
            for i in range(featureX.shape[0]):
                    x_rowvector = X[i, :]
                    deltax = (np.log(probabilty)
                              -0.5*np.log(np.linalg.det(sigma_k))
                              -0.5*x_rowvector@np.linalg.inv(sigma_k)@x_rowvector.T
                              +x_rowvector@np.linalg.inv(sigma_k)@mean_vector
                              -0.5*mean_vector.T@np.linalg.inv(sigma_k)@mean_vector)
                    delta[i, :] = deltax
            deltaX = np.hstack([deltaX, delta])

        deltaX = deltaX[:, 1:]
        classificationResults = np.argmax(deltaX, axis=1)
        # maximize the delta over k
        newX['LDAClassification'] = classificationResults.reshape(-1, 1)
    else:
        print('Pleasre make sure that featured X and mean vector\
              have the same covariate variables')
        sys.exist(0)

    return(classificationResults, newX)


q_pro, q_mean, q_sigma = fitQDA(trainX, trainY)

q_preictedY, q_newX = classifyQDA(trainX, q_pro, q_mean, q_sigma)

# ###########################################################################
# % Task 5 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# ############################################################################
Qloss = computeLoss(trainY, q_preictedY)
print(Qloss)  # [0.17757009]
Qtypeloss = computeLoss(trainY, q_preictedY, True)
print(Qtypeloss)


# ############################################################################
# % Task 6 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the mixed distribution that you
# % obtain as a result from the 'fitLDA' function.
# % Add the grid.
# % Add descriptive legend and title.
# ############################################################################
mu_1 = np.array(q_mean)[0,0]
var_1 = q_sigma['class0'][0,0]**0.5
min_1 = np.array(features_Class_1).min()
max_1 = np.array(features_Class_1).max()
mu_3 = np.array(q_mean)[1,0]
var_3 = q_sigma['class1'][0,0]**0.5
min_3 = np.array(features_Class_3).min()
max_3 = np.array(features_Class_3).max()
mu, cov = stats.norm.fit(np.array(X))
min_ = np.array(X).min()
max_ = np.array(X).max()

plt.style.use('seaborn-deep')
plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 6, QDA, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


# #########################################################################
# % Task 7 : Construct Naive Bayes Gauss classifier. For that fill in the
# % function fitNaiveBayesGauss and classifyNaiveBayesGauss.
# % Both functions should be constructed in order to work with multiple
# % classes and multiple features if needed. However, we start with only
# % two-classes classification.
# #########################################################################
def fitNBG(X, Y):
    '''
    THIS FUNCTION IS EXACTLY SAME with fitQDA
    X and Y are dataframe
    need to be transfomed into:
    Input: X - n by m matrix, containing all features
           Y - n by 1 matrix, contanning the information of class
    Causion: the orders of rows of X and Y are assumed to be corresponding to
             each other, which means inforamtion of any row (say row 10) of X
             and information of same row (row 10) of Y are from same
             observation
    Output: prior_probability, mean and covariance
    '''
    n = X.shape[0]  # get the number of total observation
    columnsIndex = X.columns
    m = X.shape[1]  # get the number of covariate variables (features)
    X = np.asmatrix(X).reshape(n, m)
    Y = np.asarray(Y).reshape(-1, 1)  # Y has to be an array, not matrix
    yunique = np.unique(Y)
    prior_probability = {}  # initialize a distionary for prior probability
    mu_k = {}
    sigma = {}

    for i in yunique:
        rowindex = np.where(Y == i)[0]
        classlabel = 'class' + str(i)
        prior_probability[classlabel] = len(rowindex)/n
        filteredx = X[rowindex, :]
        mean_vector = np.mean(filteredx, axis=0)
        mu_k[classlabel] = mean_vector
        mean_maxtrix = np.repeat(mean_vector, filteredx.shape[0], axis=0)
        diff_matrix = filteredx - mean_maxtrix
        sigma_k = (diff_matrix.T @ diff_matrix)/(n-1)
        # calcluate within-class covariance
        sigma[classlabel] = sigma_k

    # tidy the output
    prior_probability = pd.DataFrame(list(prior_probability.values()),
                                     index=prior_probability.keys(),
                                     columns=['Prior Probability'])
    mean_dataframe = []
    for v in mu_k:
        mean_dataframe.extend(np.array(mu_k[v]))
    mu_k = pd.DataFrame(mean_dataframe, index=mu_k.keys(),
                        columns=columnsIndex)

    return(prior_probability, mu_k, sigma)


def classifyNBG(featureX, priorpro, mu, sigma):
    '''
    Same Input, same out
    But algorithm is different, we need employ the pdf of Gaussian Normal
    '''
    # calcluate probability from Gaussian Nomral
    newX = pd.DataFrame.copy(featureX)
    classLabels = priorpro.index  # get class labes from dataframe
    featureLabels = featureX.columns
    meanLabels = mu.columns
    X = np.asmatrix(featureX)
    m = featureX.shape[1]
    delta = np.zeros([featureX.shape[0], 1])
    deltaX = np.zeros([featureX.shape[0], 1])
    if all(featureLabels == meanLabels):
        for v in classLabels:
            probabilty = np.array(priorpro.loc[v, :])
            mean_vector = np.array(mu.loc[v, :]).reshape(1, -1)
            sigma_k = sigma[v]
            for i in range(featureX.shape[0]):
                x_rowvector = X[i, :]
                x_diff = (x_rowvector - mean_vector).reshape(1, -1)
                zmod = np.sqrt(np.power((2*math.pi),m)
                               *np.linalg.det(sigma_k))
                post_prob = 1/zmod*np.exp(-0.5*x_diff@np.linalg.inv(sigma_k)@x_diff.T)
                delta[i, :] = post_prob * probabilty

            deltaX = np.hstack([deltaX, delta])

        deltaX = deltaX[:, 1:]
        deltaX = np.divide(deltaX, np.sum(deltaX, axis=1).reshape(-1, 1))
        classificationResults = np.argmax(deltaX, axis=1)
        # maximize the delta over k
        newX['LDAClassification'] = classificationResults.reshape(-1, 1)

    else:
        print('Pleasre make sure that featured X and mean vector\
              have the same covariate variables')
        sys.exist(0)

    return(classificationResults, newX)


ngb_pro, ngb_mean, ngb_sigma = fitNBG(trainX, trainY)

ngb_preictedY, ngb_newX = classifyNBG(trainX, ngb_pro, ngb_mean, ngb_sigma )


# #########################################################################
# % Task 8 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# #########################################################################
ngbloss = computeLoss(trainY, ngb_preictedY)
print(ngbloss)  # [0.17757009]
ngbpeloss = computeLoss(trainY, ngb_preictedY, True)
print(ngbpeloss)

# #########################################################################
# % Task 9 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the mixed distribution that you
# % obtain as a result from the 'fitNaiveBayesGauss' function.
# % Add the grid.
# % Add descriptive legend and title.
# #########################################################################
mu_1 = np.array(ngb_mean)[0,0]
var_1 = ngb_sigma['class0'][0,0]**0.5
min_1 = np.array(features_Class_1).min()
max_1 = np.array(features_Class_1).max()
mu_3 = np.array(ngb_mean)[1,0]
var_3 = ngb_sigma['class1'][0,0]**0.5
min_3 = np.array(features_Class_3).min()
max_3 = np.array(features_Class_3).max()
mu, cov = stats.norm.fit(np.array(X))
min_ = np.array(X).min()
max_ = np.array(X).max()

plt.style.use('seaborn-deep')
plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 9, NBG, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()


# Part III: Binary Classification with Two Features ======================
#
# % Select only classes 1 and 3 for this part and features:
# %
# %   - 'Proanthocyanins'
# %   - 'Alcalinity of ash'
# %
# % In this binary classification exercise assign label 0 to Class 1 and
# % label 1 to Class 3.

part3wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part3wine = part3wine[['WineClass', 'Proanthocyanins','Alcalinity of ash']]
part3wine['newclass'] = np.where(part3wine['WineClass'] == 1, 0, 1)
part3wine = part3wine[['newclass', 'Proanthocyanins', 'Alcalinity of ash']]
part3wine.shape
part3wine.head()


# #########################################################################
# % Task 10 : Construct Naive Bayes Gauss classifier.
# #########################################################################

trainX3 = part3wine[['Proanthocyanins', 'Alcalinity of ash']]
trainY3 = part3wine[['newclass']]

ngb_pro3, ngb_mean3, ngb_sigma3 = fitNBG(trainX3, trainY3)

ngb_preictedY3, ngb_newX3 = classifyNBG(trainX3, ngb_pro3,
                                        ngb_mean3, ngb_sigma3)


# #########################################################################
# % Task 8 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# #########################################################################
ngbloss3 = computeLoss(trainY3, ngb_preictedY3)
print(ngbloss3)  # [0.07476636]
ngbpeloss3 = computeLoss(trainY3, ngb_preictedY3, True)
print(ngbpeloss3)


# #########################################################################
# % Plot the resulting classification.
# % Add the grid.
# % Add descriptive legend and title.
# % Mark misclassified observations.
# ##########################################################################
# Proanthocyanins
X3 = np.asmatrix(part3wine [['Proanthocyanins', 'Alcalinity of ash']]).reshape(-1, 2)
Y3 = np.asmatrix(part3wine.newclass).reshape(-1, 1)
features_Class_1 = np.array(wineSubset.query('newclass==0')['Proanthocyanins'])
features_Class_3 = np.array(wineSubset.query('newclass==1')['Proanthocyanins'])

mu_1 = np.array(ngb_mean3['Proanthocyanins'])[0]
var_1 = ngb_sigma3['class0'][0,0]**0.5
min_1 = np.array(features_Class_1).min()
max_1 = np.array(features_Class_1).max()
mu_3 = np.array(ngb_mean3['Proanthocyanins'])[1]
var_3 = ngb_sigma3['class1'][0,0]**0.5
min_3 = np.array(features_Class_3).min()
max_3 = np.array(features_Class_3).max()
mu, cov = stats.norm.fit(np.array(X3[:,0]))
min_ = np.array(X).min()
max_ = np.array(X).max()

plt.style.use('seaborn-deep')
plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 8, NGB, Wine Dataset - Wine 1 and 3 Proanthocyanins Content')
plt.show()

# Alcalinity of ash
wineSubset_ = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
wineSubset_ = wineSubset_[['WineClass', 'Alcalinity of ash']]
wineSubset_['newclass'] = np.where(wineSubset_['WineClass'] == 1, 0, 1)
wineSubset_ = wineSubset_[['newclass', 'Alcalinity of ash']]
wineSubset_.shape
wineSubset_.head()
features_Class_1 = np.array(wineSubset_.query('newclass==0')['Alcalinity of ash'])
features_Class_3 = np.array(wineSubset_.query('newclass==1')['Alcalinity of ash'])

mu_1 = np.array(ngb_mean3['Alcalinity of ash'])[0]
var_1 = ngb_sigma3['class0'][1,1]**0.5
min_1 = np.array(features_Class_1).min()
max_1 = np.array(features_Class_1).max()
mu_3 = np.array(ngb_mean3['Alcalinity of ash'])[1]
var_3 = ngb_sigma3['class1'][1,1]**0.5
min_3 = np.array(features_Class_3).min()
max_3 = np.array(features_Class_3).max()
mu, cov = stats.norm.fit(np.array(X3[:,1]))
min_ = np.array(X).min()
max_ = np.array(X).max()

plt.style.use('seaborn-deep')
plt.hist(features_Class_1,density=True,label='Class 1',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_3,density=True,label='Class 3',alpha = 0.7,
         bins=np.arange(min(features_Class_3)-0.3,
                        max(features_Class_3)+0.3,0.3))
x_1 = np.linspace(min_1-0.3, max_1, 100)
plt.plot(x_1, stats.norm.pdf(x_1, mu_1, var_1),label='Class 1, normal')
x_3 = np.linspace(min_3-0.3, max_3, 100)
plt.plot(x_3, stats.norm.pdf(x_3, mu_3, var_3),label='Class 3, normal')
x = np.linspace(min_-0.3, max_, 100)
plt.plot(x, stats.norm.pdf(x, mu, cov),label='Mixture normal')
plt.grid()
plt.legend(loc='upper right')
plt.title('Task 8, NGB, Wine Dataset - Wine 1 and 3 Alcalinity of ash')
plt.show()


#  Part IV: Binary Classification with Many Features ======================
#
#
# % Select only classes 1 and 3 for this part and features:
# %
# %   - 'Alcohol'
# %   - 'Flavanoids'
# %   - 'Proanthocyanins'
# %   - 'Color intensity'
# %
# % In this binary classification exercise assign label 0 to Class 1 and
# % label 1 to Class 3.

part4wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part4wine = part4wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part4wine['newclass'] = np.where(part4wine['WineClass'] == 1, 0, 1)
part4wine = part4wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part4wine.shape
part4wine.head()


# ##########################################################################
# % Task 11 : Construct LDA classifier.
# ##########################################################################
#
#
# ##########################################################################
# % Compute the empirical value of the error using the 0-1 loss.
# ##########################################################################
#
X4 = np.asmatrix(part4wine.iloc[:, 1:]).reshape(-1, 4)
Y4 = np.asmatrix(part4wine.newclass).reshape(-1, 1)

trainX4 = part4wine[['Alcohol','Flavanoids',
                     'Proanthocyanins','Color intensity']]
trainY4 = part4wine[['newclass']]


binary_pro4, binary_mean4, binary_sigma4 = fitLDA(trainX4, trainY4)

preictedY4, newX4 = classifyLDA(trainX4, binary_pro4, binary_mean4,
                                binary_sigma4, False)

# % Task 2 : Compute the empirical value of the error using the 0-1 loss.
# % For that add typeOfLoss '0-1' option to the function computeLoss from the
# % previous assignment. Additionally, this function needs to output the Type I and Type II
# % errors (false positive and false negative) which will be filled in only in the case of
# % binary classification.

loss4 = computeLoss(trainY4, preictedY4)
print(loss4)  #[0.]
typeloss4 = computeLoss(trainY4, preictedY4, True)
print(typeloss4)

#  Part V: 3-Classes Classification with Many Features ====================
#
# % Select only classes 1, 2 and 3 for this part and features:
# %
# %   - 'Alcohol'
# %   - 'Flavanoids'
# %   - 'Proanthocyanins'
# %   - 'Color intensit

part5wine = wine[['WineClass', 'Alcohol', 'Flavanoids',
                  'Proanthocyanins', 'Color intensity']]
part5wine.shape
part5wine.head()

X5 = np.asmatrix(part5wine.iloc[:, 1:]).reshape(-1, 4)
Y5 = np.asmatrix(part5wine.WineClass).reshape(-1, 1)

trainX5 = part5wine[['Alcohol', 'Flavanoids',
                     'Proanthocyanins', 'Color intensity']]
trainY5 = part5wine[['WineClass']]
# ##########################################################################
# % Task 12 : Construct QDA classifier for the following:
# %   - 'Alcohol'
# %   - 'Alcohol' + 'Proanthocyanins'
# %   - All features listed above
#
# % Compute the empirical value of the errors using the 0-1 loss.
# ##########################################################################
q_pro5, q_mean5, q_sigma5 = fitQDA(trainX5, trainY5)

q_preictedY5, q_newX5 = classifyQDA(trainX5, q_pro5, q_mean5, q_sigma5)

loss5 = computeLoss(trainY5, q_preictedY5+1)
print(loss5)  # [0.04494382]
typeloss5 = computeLoss(trainY5, q_preictedY5+1, True)
print(typeloss5)

# End of code

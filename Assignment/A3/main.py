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

# check working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A3')

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

###############################################################################
# Part II: Binary Classification with One Feature
###############################################################################

#  Task 0
#  Plot the data by creating two <count> density-normalized histograms
#  in two different colors of your choice; for that use the specific
#  normalization and 'BinWidth' set to 0.3. Add the grid.
#  Add descriptive legend and title.

fig, axes = plt.subplots(1, 1, figsize=(8, 6))
sns.distplot(wineSubset.query('newclass==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes)
sns.distplot(wineSubset.query('newclass==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes)
axes.grid()
axes.legend()
plt.show()

# Task 1 :
# Construct LDA classifier. For that fill in the function fitLDA
# and classifyLDA. Both functions should be constructed in order to
# work with multiple classes and multiple feautures if needed. We start
# here however with only two-classes classification which admits the
# explicit critical decision boundary value.


# Function to split the sample
def splitSample(sampleX, sampleY, trainSize, permute=False):
    '''
    Input:
         sampleX: n by k dataframe, k is the number of covariate variables
         sampleY: n by 1 dataframe, in this case, it contains
            the information for classes
         trainSize: size for tranning dataseet
    '''
    sample_length = int(sampleX.shape[0] * trainSize)
    if permute is True:
        random_index = random.sample(range(sampleX.shape[0]),
                                     sample_length)
        trainSampleX = sampleX[random_index, :]
        trainSampleY = sampleY[random_index, :]
        testSampleX = np.delete(sampleX, random_index, 0)
        testSampleY = np.delete(sampleY, random_index, 0)

        return(trainSampleX, testSampleX, trainSampleY, testSampleY)
    else:
        percentile_index = list(range(sample_length))
        trainSampleX = sampleX[percentile_index, :]
        trainSampleY = sampleY[percentile_index, :]
        testSampleX = np.delete(sampleX, percentile_index, 0)
        testSampleY = np.delete(sampleY, percentile_index, 0)

        return(trainSampleX, testSampleX, trainSampleY, testSampleY)


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
    sigma = sum(sigma.values())/(n - len(yunique))  # estimate final sigma
    prior_probability = pd.DataFrame(prior_probability.values(),
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
    if all(featureLabels == meanLabels):
        delta = np.zeros([featureX.shape[0], 1])
        for v in classLabels:
            probabilty = np.array(priorpro.loc[v, :]).reshape(-1, 1)
            # get prior probabilty for class k
            mean_vector = np.array(mu.loc[v, :]).reshape(-1, 1)
            # get mean vector for class k
            deltaX = (X @ np.linalg.inv(sigma) @ mean_vector
                      - 1/2 * mean_vector.T @
                      np.linalg.inv(sigma) @ mean_vector
                      + np.log(probabilty))
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
            prob_i = np.array(priorpro.iloc[0, :]).reshape(-1, 1)
            prob_j = np.array(priorpro.iloc[1, :]).reshape(-1, 1)
            xcritical = sigma/(mean_j - mean_i)*(
                np.log(prob_i/prob_j) + (mean_j**2
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


# test LDA
X = np.asmatrix(wineSubset.Proanthocyanins).reshape(-1, 1)
Y = np.asmatrix(wineSubset.newclass).reshape(-1, 1)
trainX, testX, trainY, testY = splitSample(X, Y, 0.8, True)

trainX = pd.DataFrame(trainX, columns=['Proanthocyanins'])
trainY = pd.DataFrame(trainY, columns=['newclass'])
testX = pd.DataFrame(testX, columns=['Proanthocyanins'])
testY = pd.DataFrame(testY, columns=['newclass'])


binary_pro, binary_mean, binary_sigma = fitLDA(trainX, trainY)

preictedY, newX, critX = classifyLDA(testX, binary_pro, binary_mean, binary_sigma,
                              critical=True)
#
# % Task 2 : Compute the empirical value of the error using the 0-1 loss.
# % For that add typeOfLoss '0-1' option to the function computeLoss from the
# % previous assignment. Additionally, this function needs to output the Type I and Type II
# % errors (false positive and false negative) which will be filled in only in the case of
# % binary classification.

loss = computeLoss(testY, preictedY)
print(loss)  #[0.22727273]
typeloss = computeLoss(testY, preictedY, True)
print(typeloss)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 3 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the Gaussian mixture distribution that you
# % obtain with the parameters computed in the 'fitLDA' function.
# % Add the grid.
# % Add descriptive legend and title.
# % Plot the decision boundary (critical value for the given threshold of interest,
# % which is set by default to 1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testDataset = pd.concat([testX, testY], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
sns.distplot(testDataset.query('newclass==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[0])
sns.distplot(testDataset.query('newclass==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[0])
axes[0].axvline(critX, color='r', label='Critical X')
axes[0].grid()
axes[0].legend()
axes[0].set(title='Histogram for Original Data')
sns.distplot(newX.query('LDAClassification==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[1])
sns.distplot(newX.query('LDAClassification==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[1])
axes[1].axvline(critX, color='r', label='Critical X')
axes[1].grid()
axes[1].legend()
axes[1].set(title='Histogram for Predicted Data (LDA)')
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 4 : Construct QDA classifier. For that fill in the function fitQDA
# % and classifyQDA. Both functions should be constructed in order to
# % work with multiple classes and multiple features if needed. We start
# % here however with only two-classes classification first.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    prior_probability = pd.DataFrame(prior_probability.values(),
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

q_preictedY, q_newX = classifyQDA(testX, q_pro, q_mean, q_sigma)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 5 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Qloss = computeLoss(testY, q_preictedY)
print(Qloss)  # [0.22727273]
Qtypeloss = computeLoss(testY, q_preictedY, True)
print(Qtypeloss)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 6 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the mixed distribution that you
# % obtain as a result from the 'fitLDA' function.
# % Add the grid.
# % Add descriptive legend and title.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
sns.distplot(testDataset.query('newclass==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[0])
sns.distplot(testDataset.query('newclass==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[0])
axes[0].grid()
axes[0].legend()
axes[0].set(title='Histogram for Original Data')
sns.distplot(q_newX.query('LDAClassification==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[1])
sns.distplot(q_newX.query('LDAClassification==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[1])
axes[1].grid()
axes[1].legend()
axes[1].set(title='Histogram for Predicted Data (QDA)')
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 7 : Construct Naive Bayes Gauss classifier. For that fill in the
# % function fitNaiveBayesGauss and classifyNaiveBayesGauss.
# % Both functions should be constructed in order to work with multiple
# % classes and multiple features if needed. However, we start with only
# % two-classes classification.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
    prior_probability = pd.DataFrame(prior_probability.values(),
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

ngb_preictedY, ngb_newX = classifyNBG(testX, ngb_pro, ngb_mean, ngb_sigma )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 8 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ngbloss = computeLoss(testY, ngb_preictedY)
print(ngbloss)  # [0.22727273]
ngbpeloss = computeLoss(testY, ngb_preictedY, True)
print(ngbpeloss)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 9 : Plot the resulting classification.
# % Create two histograms in two different colors  of your choice: for these,
# % use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
# % Superimpose the two normal distributions and the mixed distribution that you
# % obtain as a result from the 'fitNaiveBayesGauss' function.
# % Add the grid.
# % Add descriptive legend and title.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fig, axes = plt.subplots(1, 2, figsize=(15, 8))
sns.distplot(testDataset.query('newclass==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[0])
sns.distplot(testDataset.query('newclass==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[0])
axes[0].grid()
axes[0].legend()
axes[0].set(title='Histogram for Original Data')
sns.distplot(ngb_newX.query('LDAClassification==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[1])
sns.distplot(ngb_newX.query('LDAClassification==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[1])
axes[1].grid()
axes[1].legend()
axes[1].set(title='Histogram for Predicted Data (NGB)')
plt.show()


# %% Part III: Binary Classification with Two Features ======================
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 10 : Construct Naive Bayes Gauss classifier.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X3 = np.asmatrix(part3wine [['Proanthocyanins', 'Alcalinity of ash']]).reshape(-1, 2)
Y3 = np.asmatrix(part3wine.newclass).reshape(-1, 1)
trainX3, testX3, trainY3, testY3 = splitSample(X3, Y3, 0.8, True)

trainX3 = pd.DataFrame(trainX3, columns=['Proanthocyanins',
                                         'Alcalinity of ash'])
trainY3 = pd.DataFrame(trainY3, columns=['newclass'])
testX3 = pd.DataFrame(testX3, columns=['Proanthocyanins',
                                       'Alcalinity of ash'])
testY3 = pd.DataFrame(testY3, columns=['newclass'])


ngb_pro3, ngb_mean3, ngb_sigma3 = fitNBG(trainX3, trainY3)

ngb_preictedY3, ngb_newX3 = classifyNBG(testX3, ngb_pro3, ngb_mean3, ngb_sigma3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 8 : Compute the empirical value of the error using the 0-1 loss.
# % using the function computeLoss together with the Type I and Type II
# % errors
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ngbloss3 = computeLoss(testY3, ngb_preictedY3)
print(ngbloss3)  # [0.04545455]
ngbpeloss3 = computeLoss(testY3, ngb_preictedY3, True)
print(ngbpeloss3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Plot the resulting classification.
# % Add the grid.
# % Add descriptive legend and title.
# % Mark misclassified observations.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testDataset3 = pd.concat([testX3, testY3], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.distplot(testDataset3.query('newclass==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[0,0])
sns.distplot(testDataset3.query('newclass==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[0,0])
axes[0,0].grid()
axes[0,0].legend()
axes[0,0].set(title='Histogram for Original Data')
sns.distplot(ngb_newX3.query('LDAClassification==0')['Proanthocyanins'],
             bins=30, hist=True, color="#4688F1", label="Class 0", ax=axes[0,1])
sns.distplot(ngb_newX3.query('LDAClassification==1')['Proanthocyanins'],
             bins=30, hist=True, color="#6639B6", label="Class 1", ax=axes[0,1])
axes[0,1].grid()
axes[0,1].legend()
axes[0,1].set(title='Histogram for Predicted Data (NGB)')
sns.distplot(testDataset3.query('newclass==0')['Alcalinity of ash'],
             bins=30, hist=True, color="#F34235", label="Class 0", ax=axes[1,0])
sns.distplot(testDataset3.query('newclass==1')['Alcalinity of ash'],
             bins=30, hist=True, color="#4BAE4F", label="Class 1", ax=axes[1,0])
axes[1,0].grid()
axes[1,0].legend()
axes[1,0].set(title='Histogram for Original Data')
sns.distplot(ngb_newX3.query('LDAClassification==0')['Alcalinity of ash'],
             bins=30, hist=True, color="#F34235", label="Class 0", ax=axes[1,1])
sns.distplot(ngb_newX3.query('LDAClassification==1')['Alcalinity of ash'],
             bins=30, hist=True, color="#4BAE4F", label="Class 1", ax=axes[1,1])
axes[1,1].grid()
axes[1,1].legend()
axes[1,1].set(title='Histogram for Predicted Data (NGB)')
plt.show()
#
# %% Part IV: Binary Classification with Many Features ======================
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 11 : Construct LDA classifier.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Compute the empirical value of the error using the 0-1 loss.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
X4 = np.asmatrix(part4wine.iloc[:, 1:]).reshape(-1, 4)
Y4 = np.asmatrix(part4wine.newclass).reshape(-1, 1)
trainX4, testX4, trainY4, testY4 = splitSample(X4, Y4, 0.8, True)

trainX4 = pd.DataFrame(trainX4, columns=['Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity'])
trainY4 = pd.DataFrame(trainY4, columns=['newclass'])
testX4 = pd.DataFrame(testX4, columns=['Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity'])
testY4 = pd.DataFrame(testY4, columns=['newclass'])




binary_pro4, binary_mean4, binary_sigma4 = fitLDA(trainX4, trainY4)

preictedY4, newX4 = classifyLDA(testX4, binary_pro4, binary_mean4,
                                binary_sigma4, False)
#
# % Task 2 : Compute the empirical value of the error using the 0-1 loss.
# % For that add typeOfLoss '0-1' option to the function computeLoss from the
# % previous assignment. Additionally, this function needs to output the Type I and Type II
# % errors (false positive and false negative) which will be filled in only in the case of
# % binary classification.

loss4 = computeLoss(testY4, preictedY4)
print(loss4)  #[0.]
typeloss4 = computeLoss(testY4, preictedY4, True)
print(typeloss4)

# %% Part V: 3-Classes Classification with Many Features ====================
#
# % Select only classes 1, 2 and 3 for this part and features:
# %
# %   - 'Alcohol'
# %   - 'Flavanoids'
# %   - 'Proanthocyanins'
# %   - 'Color intensit

part5wine = wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part5wine.shape
part5wine.head()

X5 = np.asmatrix(part5wine.iloc[:, 1:]).reshape(-1, 4)
Y5 = np.asmatrix(part5wine.WineClass).reshape(-1, 1)
trainX5, testX5, trainY5, testY5 = splitSample(X5, Y5, 0.8, True)

trainX5 = pd.DataFrame(trainX5, columns=['Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity'])
trainY5 = pd.DataFrame(trainY5, columns=['WineClass'])
testX5 = pd.DataFrame(testX5, columns=['Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity'])
testY5 = pd.DataFrame(testY5, columns=['WineClass'])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Task 17 : Construct QDA classifier for the following:
# %
# %   - 'Alcohol'
# %   - 'Alcohol' + 'Proanthocyanins'
# %   - All features listed above
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Compute the empirical value of the errors using the 0-1 loss.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#

q_pro5, q_mean5, q_sigma5 = fitQDA(trainX5, trainY5)

q_preictedY5, q_newX5 = classifyQDA(testX5, q_pro5, q_mean5, q_sigma5)

loss5 = computeLoss(testY5, q_preictedY5+1)
print(loss5)  #  [0.05555556]
typeloss5 = computeLoss(testY5, q_preictedY5+1, True)
print(typeloss5)


# End of code

# Assignment 4 - Classification
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
from sklearn.linear_model import LogisticRegression


# check working directory
os.getcwd()
# os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A4')
# os.chdir('C:/Users/User/Desktop/ML/assignments/3')

###############################################################################
# Part 0 - Load the Dataset
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

###############################################################################
# Part I.0 - Employ the Class to do the task I
###############################################################################

class LP_classify(object):
    """
    A class to do the classification.
    supervised: logistic and perceptron
    The standard procedure to do classfication in Machine Learning
    Step 1: Train the Data
            1-a): split the sample into the trainning sample and test sample
            1-b): decide whether normalize or standardize dataset
            1-c): chose the method to train the algorithm
                  1-c-i): logistic classficiation
                  1-c-ii): perceptron classification
    Step 2: test the algorithm
            2-a): calcuate the loss or accuracy
            2-b): calcuate the cross-valiation error

    The structure of Class LKP_classify(args, kwargs):

    S1-Initialize the class and split the sample:
        Input: matrix X: n by m
               vector Y: n by 1
               normalize or standardize data
    S2-methods: logistic, perceptron
    S3-prediction and calcuate the loss
    """
    def __init__(self, X, Y, percentile, randomsplit=False,
                 normalized=False, standardized=False, constant=False):
        '''
        Initilize the input: (order matters)
        Taken X and Y as dataframe or matrix format, but still trying to
        convert to array and matrix
        Default model includes interecpet
        percential: the ratio for splitting the sample
        randomsplit: if it is true, sample is splited  randomly
        constant=False, if it is true, creat the constant vector and
        add it to X.
        '''
        self.X = X
        self.Y = Y
        self.percentile = percentile
        try:
            self.X = np.asmatrix(self.X)
            self.Y = np.asmatrix(self.Y).reshape(-1, 1)
            if (self.X.shape[0] != self.Y.shape[0]):
                print('Input Y and X \
                      have different sample size')
                sys.exit(0)
        except Exception:
            print('There is an error with the input data.\
                  Make sure input are either matrix or dataframe')
            sys.exit(0)

        # normalize or standardize
        if normalized is True and standardized is True:
            print("You can either only normalize X or standardize X,\
                  but not at the same time: dont't set them true at\
                  the same time")
            sys.exit(0)
        elif normalized is True:
            self.X = np.apply_along_axis(LP_classify.normalize,
                                         axis=0, arr=self.X)
        elif standardized is True:
            self.X = np.apply_along_axis(LP_classify.standard,
                                         axis=0, arr=self.X)
        if constant is True:
            vector_one = np.ones(self.X.shape[0]).reshape(-1, 1)
            self.X = np.hstack((vector_one, self.X))

        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            LP_classify.splitSample(self.X, self.Y, percentile, randomsplit))

    @staticmethod
    def splitSample(sampleX, sampleY, trainSize, permute=False):
        '''
        static function to split the sample
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

    # normalize function
    @staticmethod
    def normalize(array):
        arrayNorm = (array - array.min())/(array.max() - array.min())
        return arrayNorm

    # standardlize function
    @staticmethod
    def standard(array):
        arrayStand = (array - array.mean())/array.std()
        return arrayStand

    @staticmethod
    def sigmoid(z):
        gz = 1/(1+np.exp(-z))
        return(gz)

    @staticmethod
    def softmax(z):
        z = np.exp(z - np.max(z))  # prevent overflow
        return z / np.sum(z, axis=1).reshape(-1, 1)

    def fitLogistic(self, alpha, regulation, rgweight, iter):
        '''
        THIS IS ALSO WORKING FOR MULTIPLE CLASS
        The algorithm is not using one-Vs-ALL but softmax regression
        when there are two classes, softmax <=> logistic
        Input:
             self.xtrain and self.ytrain
             learning rate: alpha
             regulation: 'l1' or 'l2'
             rgweight: regulation weight (lambda)
             iter: the number of iteration
        output:
             estimated coefficients
        '''
        # get the key dimensions
        n, m = self.xtrain.shape
        k = len(np.unique(self.ytrain, axis=0))  # number of class

        # initialize the coefficients, m by k
        self.log_coefs = np.zeros([m, k])
        # dummy matrix, n by k
        y_mat = np.asmatrix(
            pd.get_dummies(np.asarray(self.ytrain).reshape(-1)).astype(float))

        # based on regulation methods to train the model
        if regulation == 'l2':
            # do the itertion
            for i in range(iter):
                randomIndex = random.sample(range(n), n)
                for j in randomIndex:
                    # calculate the gradient first
                    yhat = LP_classify.softmax(self.xtrain[j, :] @ self.log_coefs)
                    y_diff = y_mat[j, :] - yhat  # n by k
                    self.log_coefs[1:, :] += (alpha*self.xtrain[j, 1:].T @ y_diff
                                              -alpha*rgweight*self.log_coefs[1:, :])
                    self.log_coefs[:1, :] += (alpha*self.xtrain[j, :1].T@ y_diff)

        elif regulation == 'l1':
            for i in range(iter):
                # calculate the gradient first
                # yhat = LP_classify.softmax(self.xtrain @ self.log_coefs)
                # y_diff = y_mat - yhat  # n by k
                # self.log_coefs += (1/n*alpha*self.xtrain.T @ y_diff
                # -alpha *rgweight*np.sign(self.log_coefs))
                randomIndex = random.sample(range(n), n)
                for j in randomIndex:
                    # calculate the gradient first
                    yhat = LP_classify.softmax(self.xtrain[j, :] @ self.log_coefs)
                    y_diff = y_mat[j, :] - yhat  # n by k
                    self.log_coefs[1:, :] += (alpha*self.xtrain[j, 1:].T @ y_diff
                                              -alpha*rgweight*
                                              np.sign(self.log_coefs[1:, :]))
                    self.log_coefs[:1, :] += (alpha*self.xtrain[j, :1].T@ y_diff)

        else:
            print("make sure you assign 'regulation' = 'l1' or 'l2'")

    # define a prediction function
    def Logpredict(self):
        train_logpredict = LP_classify.softmax(self.xtrain @ self.log_coefs)
        self.train_logpred_labels = np.argmax(train_logpredict, axis=1)+np.min(self.Y)
        test_logpredict = LP_classify.softmax(self.xtest @ self.log_coefs)
        self.test_logpred_labels = np.argmax(test_logpredict, axis=1)+np.min(self.Y)
        self.train_logaccuracy = np.sum(self.ytrain == self.train_logpred_labels,
                                     axis=0)/self.ytrain.shape[0]
        self.train_logloss = 1 - self.train_logaccuracy
        self.test_logaccuracy = np.sum(self.ytest == self.test_logpred_labels,
                                    axis=0)/self.ytest.shape[0]
        self.test_logloss = 1 - self.test_logaccuracy


###############################################################################
# Part I.1 - train and test the model, 3 classes, 4 features
###############################################################################

X1 = np.asmatrix(wine[['Alcohol', 'Malic acid', 'Ash','Alcalinity of ash']])
X1.shape
Y1 = np.asmatrix(wine.WineClass).reshape(-1, 1)


# L2 regulation with lambda = 0, not randomly split the sample
logist1a = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=False)
logist1a.fitLogistic(alpha=0.01, regulation='l2', rgweight=0, iter=200)
logist1a.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist1a.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:', logist1a.test_logaccuracy)


# randomly split the sample
logist1b = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=True)
logist1b.fitLogistic(alpha=0.01, regulation='l2', rgweight=0, iter=600)
logist1b.Logpredict()
print('The accuracy  of prediction for tranning dataset is:', logist1b.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist1b.test_logaccuracy)


# L1 regulation with lambda = 0
logist2 = LP_classify(X1, Y1, 0.8, constant=True)
logist2.fitLogistic(alpha=0.01, regulation='l1', rgweight=0, iter=600)
logist2.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist2.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist2.test_logaccuracy)

# sklearn test
# clf = LogisticRegression(random_state=0, solver='lbfgs',
#                          multi_class='multinomial').fit(logist1.xtrain, logist1.ytrain)
# np.sum(clf.predict(logist1.xtrain) == np.asarray(logist1.ytrain).reshape(-1), axis=0)/142
# np.sum(clf.predict(logist1.xtest) == np.asarray(logist1.ytest).reshape(-1), axis=0)/36
# # in terms of accuracy, there is no big differences

# tune the regulation weight
# L2 regulation with lambda = 0.01
logist3 = LP_classify(X1, Y1, 0.8, constant=True)
logist3.fitLogistic(alpha=0.01, regulation='l2', rgweight=0.01, iter=600)
logist3.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist3.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist3.test_logaccuracy)

# Find the optimal regulation
l2_regulations = [1/(10**i) for i in range(1,7)]  # test 7 regulation parameters

# 5-fold cross-validation
# Warning: it takes around 5 miniutes
kfold = 5
test_perform = {} # initialize a dict to store performance

for lk in l2_regulations:
    # split the sample
    nrows = X1.shape[0]
    subsetRows = math.floor(nrows/kfold)
    k_randomIndex = random.sample(range(nrows), nrows)

    kfold_perf = {}
    for i in range(kfold):
        subsetIndex = k_randomIndex[i*subsetRows:(i+1)*subsetRows]
        testsubset_x = X1[subsetIndex, :]
        trainsubset_x = np.delete(X1, subsetIndex, 0)
        testsubset_y = Y1[subsetIndex, :]
        trainsubset_y = np.delete(Y1, subsetIndex, 0)
        logkfold = LP_classify(trainsubset_x, trainsubset_y, 1,
                               constant=True)
        logkfold.fitLogistic(alpha=0.01, regulation='l2', rgweight=lk, iter=200)
        # add constant
        X_mat = np.hstack([np.ones([testsubset_x.shape[0], 1]), testsubset_x])
        kfoldpred = LP_classify.softmax(X_mat @ logkfold.log_coefs)
        test_loglabels = np.argmax(kfoldpred, axis=1)+np.min(Y1)
        test_accuracy = np.sum(testsubset_y == test_loglabels,
                               axis=0)/testsubset_y[0]
        kfold_perf[str(i)] = np.asarray(test_accuracy).reshape(-1)

    test_perform[str(lk)] = kfold_perf

print(pd.DataFrame(test_perform))
print('The regulation number can produce the highest test accuracy is:', pd.DataFrame(test_perform).sum(axis=0)/5)
# it's 0.0001 for 600 iterations
# it 1e-6 for 200 iterations one round

# try with 0.0001 again

logist4 = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=True)
logist4.fitLogistic(alpha=0.01, regulation='l2', rgweight=1e-6, iter=200)
logist4.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist4.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist4.test_logaccuracy)
# 52%

# try with standardize the data
logist5 = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=True,
                      standardized=True)
logist5.fitLogistic(alpha=0.01, regulation='l2', rgweight=0.00001, iter=200)
logist5.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist5.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist5.test_logaccuracy)
# both of classification is close to 80%, it's qutie amazing with only 200 iterations

###############################################################################
# Part I.2 - Final test: binary classification with many features
###############################################################################
part4wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part4wine = part4wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part4wine['newclass'] = np.where(part4wine['WineClass'] == 1, 0, 1)
part4wine = part4wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part4wine.shape
part4wine.head()

X4 = np.asmatrix(part4wine.iloc[:, 1:]).reshape(-1, 4)
Y4 = np.asmatrix(part4wine.newclass).reshape(-1, 1)

logist6 = LP_classify(X4, Y4, 0.8, constant=True, randomsplit=True,
                      standardized=True)
logist6.fitLogistic(alpha=0.01, regulation='l2', rgweight=0.00001, iter=200)
logist6.Logpredict()
# logist6.ytest
# logist6.test_logpred_labels
print('The accuracy  of prediction for tranning dataset is:',logist6.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist6.test_logaccuracy)
# THE prediction accuracy is 100% !

###############################################################################
# Comment on Task 1

# As the softmax classification can be seen as one layer of deep learning,
# the tranning practices from logist1 to logist5 should lead the programmer to
# learn the following lessons:
#                       [All with SGD for softmax]

# 1) without regulation, up to 200 iteration,
#         the prediction accuracy can increase to 70% for tainning dataset;
#         however, the accuracy is only around 30% for test dataset

# 2) without regulation, up to 600 - 800 iteration,
#         the prediction accuracy can increase to 80%,
#         however, the accuracy is still only around 30% for test dataset
# One can see that the marginal gain is not big by doing more interations
# without regulations

# 3) with regulation and optimal L2-lambda value, the prediction accuracy
#         for tranning data can increase to 70-80%, and more importantly,
#         the accuracy for test dataset can jump to 70%

# 4) with regulation and standardized dataset, the accuracy for both tranning
#    and test dataset are around 90%, which is very impressive.

# Tips for tunning the deep learning network
# i) standardize or normalize the dataset
# ii) do regulation, and find the optimal regulation l2-lambda, 0.0001,
#     0.00001, 1e-6, are all fine
# iii) try different features from X
###############################################################################

###############################################################################
# Part II.0 - Employ the Class to do the task II
###############################################################################

class Perceptron_binary(object):
    """
    This is the class to do the binary classification with percetron function
    The percetron model is very close to the logisitc regression, where
    logisitc regression employs the sigmoid function, but percetron employs
    the boundary value function assigning -1 (wx+b<0) 1 (wx+b>0)
    # the notation in this programming follows the notes by Ng, Andrew
    # http://cs229.stanford.edu/notes/cs229-notes-all/cs229-notes6.pdf
    Since we are using 1 and -1, the updating rule is slightly different now
    """
    def __init__(self, X, Y, percentile, randomsplit=False,
                 normalized=False, standardized=False, constant=False):
        '''
        Initilize the input: (order matters)
        Taken X and Y as dataframe or matrix format, but still trying to
        convert to array and matrix
        Default model includes interecpet
        percential: the ratio for splitting the sample
        randomsplit: if it is true, sample is splited  randomly
        constant=False, if it is true, creat the constant vector and
        add it to X.
        '''
        self.X = X
        self.Y = Y
        self.percentile = percentile
        try:
            self.X = np.asmatrix(self.X)
            self.Y = np.asmatrix(self.Y).reshape(-1, 1)
            if (self.X.shape[0] != self.Y.shape[0]):
                print('Input Y and X \
                      have different sample size')
                sys.exit(0)
        except Exception:
            print('There is an error with the input data.\
                  Make sure input are either matrix or dataframe')
            sys.exit(0)

        # normalize or standardize
        if normalized is True and standardized is True:
            print("You can either only normalize X or standardize X,\
                  but not at the same time: dont't set them true at\
                  the same time")
            sys.exit(0)
        elif normalized is True:
            self.X = np.apply_along_axis(LP_classify.normalize,
                                         axis=0, arr=self.X)
        elif standardized is True:
            self.X = np.apply_along_axis(LP_classify.standard,
                                         axis=0, arr=self.X)
        if constant is True:
            vector_one = np.ones(self.X.shape[0]).reshape(-1, 1)
            self.X = np.hstack((vector_one, self.X))

        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            LP_classify.splitSample(self.X, self.Y, percentile, randomsplit))

    @staticmethod
    def splitSample(sampleX, sampleY, trainSize, permute=False):
        '''
        static function to split the sample
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

    # normalize function
    @staticmethod
    def normalize(array):
        arrayNorm = (array - array.min())/(array.max() - array.min())
        return arrayNorm

    # standardlize function
    @staticmethod
    def standard(array):
        arrayStand = (array - array.mean())/array.std()
        return arrayStand

    @staticmethod
    def signfun(z):
        '''
        Input: a vector with shape = n by 1
        '''
        gz = np.ones([z.shape[0], 1])
        for i in range(z.shape[0]):
            if z[i, 0] < 0:
                gz[i, 0] = -1
            else:
                gz[i, 0] = 1
        return(gz)

    def fitPerceptron(self, alpha, iter):
        '''
        Input:
            alpha - learning rate, one can drop alpha as it does not
            affect the estimation;
            iter - the maximum iteration number
        '''
        # get the key dimensions
        n, m = self.xtrain.shape

        # initialize the coefficients, m by 1
        self.perctr_coefs = np.zeros([m, 1])
        # dummy matrix, n by k

        # based on regulation methods to train the model
        for it in range(iter):
            randomIndex = random.sample(range(n), n)
            for j in randomIndex:
                y_vect = self.xtrain[j, :] @ self.perctr_coefs
                y_vect = Perceptron_binary.signfun(y_vect)  # classify
                if y_vect != self.Y[j, :]:
                    self.perctr_coefs += self.xtrain[j, :] @ y_vect

    # prediction function
    def Pct_predict(self):
        train_pctpredict = Perceptron_binary.signfun(self.xtrain @ self.perctr_coefs)
        self.pct_train_accuracy = np.sum(train_pctpredict == self.ytrain, axis=0)/self.ytrain.shape[0]
        test_pctpredict = Perceptron_binary.signfun(self.xtest @ self.perctr_coefs)
        self.pct_test_accuracy = np.sum(test_pctpredict  == self.ytest, axis=0)/self.ytest.shape[0]




























# End of Code

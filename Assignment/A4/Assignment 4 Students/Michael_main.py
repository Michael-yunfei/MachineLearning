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
from matplotlib.patches import Ellipse, Circle


# check working directory
os.getcwd()
# please change it to the directory that contains the wine.csv dataset
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
logist1b.fitLogistic(alpha=0.01, regulation='l2', rgweight=0, iter=200)
logist1b.Logpredict()
print('The accuracy  of prediction for tranning dataset is:', logist1b.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist1b.test_logaccuracy)


# L1 regulation with lambda = 0
logist2 = LP_classify(X1, Y1, 0.8, constant=True)
logist2.fitLogistic(alpha=0.01, regulation='l1', rgweight=0, iter=200)
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
logist3.fitLogistic(alpha=0.01, regulation='l2', rgweight=0.01, iter=200)
logist3.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist3.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist3.test_logaccuracy)

###############################################################################
# 5-fold cross-validation
###############################################################################
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
                               axis=0)/testsubset_y.shape[0]
        kfold_perf[str(i)] = np.asarray(test_accuracy).reshape(-1)

    test_perform[str(lk)] = kfold_perf

print(pd.DataFrame(test_perform))
print('The regulation number can produce the highest test accuracy is:', pd.DataFrame(test_perform).sum(axis=0)/5)
# it's 0.0001 for 600 iterations
# it 1e-5/1e-6 for 200 iterations one round

# try with 0.0001 again

logist4 = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=True)
logist4.fitLogistic(alpha=0.01, regulation='l2', rgweight=1e-5, iter=200)
logist4.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist4.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist4.test_logaccuracy)


# try with standardize the data
logist5 = LP_classify(X1, Y1, 0.8, constant=True, randomsplit=True,
                      standardized=True)
logist5.fitLogistic(alpha=0.01, regulation='l2', rgweight=1e-5, iter=200)
logist5.Logpredict()
print('The accuracy  of prediction for tranning dataset is:',logist5.train_logaccuracy)
print('The accuracy  of prediction for test dataset is:',logist5.test_logaccuracy)
# both of classification is close to 80%, it's qutie amazing with only 200 iterations

###############################################################################
# Part I.2 - Final test: binary classification with 4 features
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
logist6.fitLogistic(alpha=0.01, regulation='l2', rgweight=1e-5, iter=200)
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
            self.X = np.apply_along_axis(Perceptron_binary.normalize,
                                         axis=0, arr=self.X)
        elif standardized is True:
            self.X = np.apply_along_axis(Perceptron_binary.standard,
                                         axis=0, arr=self.X)
        if constant is True:
            vector_one = np.ones(self.X.shape[0]).reshape(-1, 1)
            self.X = np.hstack((vector_one, self.X))

        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            Perceptron_binary.splitSample(self.X, self.Y, percentile, randomsplit))

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
        gz = np.zeros([z.shape[0], 1])
        for i in range(z.shape[0]):
            if z[i, 0] < 0.0:
                gz[i, 0] = -1.0
            else:
                gz[i, 0] = 1.0
        return(gz)

    def fitPerceptron(self, alpha, iter, method):
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
        #
        for it in range(iter):
            if method == 'SGD':
                randomIndex = random.sample(range(n), n)
            else:
                randomIndex = range(n)
            for j in randomIndex:
                y_vect = self.xtrain[j, :] @ self.perctr_coefs
                y_vect = Perceptron_binary.signfun(y_vect).reshape(-1, 1) # classify
                if y_vect[0, :] != self.ytrain[j, :]:
                    self.perctr_coefs += alpha * self.xtrain[j, :].T @ self.ytrain[j, :]
    # prediction function
    def Pct_predict(self):
        self.train_pctpredict = Perceptron_binary.signfun(self.xtrain @ self.perctr_coefs)
        self.pct_train_accuracy = np.sum(self.train_pctpredict == self.ytrain, axis=0)/self.ytrain.shape[0]
        self.test_pctpredict = Perceptron_binary.signfun(self.xtest @ self.perctr_coefs)
        self.pct_test_accuracy = np.sum(self.test_pctpredict  == self.ytest, axis=0)/self.ytest.shape[0]


###############################################################################
# Part II.1 - train the model, binary with 4 feature test
###############################################################################
# binary with 4 features test
part5wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part5wine = part5wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part5wine['newclass'] = np.where(part5wine['WineClass'] == 1, 1, -1)
part5wine = part5wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part5wine.shape
part5wine.head()

X5 = np.asmatrix(part5wine.iloc[:, 1:]).reshape(-1, 4)
Y5 = np.asmatrix(part5wine.newclass).reshape(-1, 1)

perceptron1 = Perceptron_binary(X5, Y5, 0.8, randomsplit=False,
                                constant=True)
perceptron1.fitPerceptron(1, 3, 'SGD')
perceptron1.Pct_predict()
print(perceptron1.pct_train_accuracy)  # 100%
print(perceptron1.pct_test_accuracy)  # 100%

# It's quite amazing that only 3 iterations, it already got 100% accuracy

perceptron1.fitPerceptron(1, 15, 'GD')
perceptron1.Pct_predict()
print(perceptron1.pct_train_accuracy)  # matrix([[0.30588235]])
print(perceptron1.pct_test_accuracy)

# Calculate the update K
# K = 16
perceptron1.fitPerceptron(1, 16, 'GD')
perceptron1.Pct_predict()
print(perceptron1.pct_train_accuracy)
print(perceptron1.pct_test_accuracy)
# [[1.]]
# [[0.95454545]]

# Error
print(1 -perceptron1.pct_train_accuracy)
print(1 - perceptron1.pct_test_accuracy)

###############################################################################
# Part II.2 - train the model, binary with 2 feature test and plot
###############################################################################

part6wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part6wine = part6wine[['WineClass', 'Proanthocyanins','Alcalinity of ash']]
part6wine['newclass'] = np.where(part6wine['WineClass'] == 1, 1, -1)
part6wine = part6wine[['newclass', 'Proanthocyanins', 'Alcalinity of ash']]
part6wine.shape
part6wine.head()

X6 = np.asmatrix(part6wine.iloc[:, 1:]).reshape(-1, 2)
Y6 = np.asmatrix(part6wine.newclass).reshape(-1, 1)

# plot the dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(part6wine.Proanthocyanins[part6wine.newclass==1],
           part6wine['Alcalinity of ash'][part6wine.newclass==1],
           facecolors='#FFFD38', edgecolors='grey',
           s=60, label='Class 1')
ax.scatter(part6wine.Proanthocyanins[part6wine.newclass==-1],
           part6wine['Alcalinity of ash'][part6wine.newclass==-1],
           marker='+', c='k', s=60, linewidth=2,
           label='Class -1')
ax.set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
ax.legend(frameon=True, fancybox=True)
plt.show()

# the dataset
perceptron2 = Perceptron_binary(X6, Y6, 0.8, randomsplit=True,
                                constant=True)
perceptron2.fitPerceptron(1, 30, 'SGD')
perceptron2.Pct_predict()
print(perceptron2.pct_train_accuracy)
print(perceptron2.pct_test_accuracy)


# plot the Classification
perct_train_idx1 = np.where(perceptron2.ytrain==1)[0]
perct_train_idx1a = np.where(perceptron2.train_pctpredict==1)[0]
perct_test_idx1 = np.where(perceptron2.ytest==1)[0]
perct_test_idx1a = np.where(perceptron2.test_pctpredict==1)[0]
perct_train_idx2 = np.where(perceptron2.ytrain==-1)[0]
perct_train_idx2a = np.where(perceptron2.train_pctpredict==-1)[0]
perct_test_idx2 = np.where(perceptron2.ytest==-1)[0]
perct_test_idx2a = np.where(perceptron2.test_pctpredict==-1)[0]

# all together plot
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes[0, 0].scatter(np.asarray(perceptron2.xtrain[perct_train_idx1,1:2]),
                np.asarray(perceptron2.xtrain[perct_train_idx1,2:3]),
                facecolors='#FFFD38', edgecolors='grey',
                s=60, label='Ground True Class 1 (train)')
axes[0, 0].scatter(np.asarray(perceptron2.xtrain[perct_train_idx2,1:2]),
                np.asarray(perceptron2.xtrain[perct_train_idx2,2:3]),
                marker='+', c='k', s=60, linewidth=2,
                label='Ground True Class -1 (train)')
axes[0, 0].add_artist(Ellipse((1, 23), 1, 9, color='b', alpha=0.2))
axes[0, 0].add_artist(Ellipse((2.2, 17), 1.7, 9, color='g', alpha=0.2))
axes[0, 1].scatter(np.asarray(perceptron2.xtrain[perct_train_idx1a,1:2]),
                np.asarray(perceptron2.xtrain[perct_train_idx1a,2:3]),
                facecolors='#FFFD38', edgecolors='grey',
                s=60, label='Estimated Class 1 (train)')
axes[0, 1].scatter(np.asarray(perceptron2.xtrain[perct_train_idx2a,1:2]),
                np.asarray(perceptron2.xtrain[perct_train_idx2a,2:3]),
                marker='+', c='k', s=60, linewidth=2,
                label='Estimated Class -1 (train)')
axes[0, 1].add_artist(Ellipse((1, 23), 1, 9, color='b', alpha=0.2))
axes[0, 1].add_artist(Ellipse((2.2, 17), 1.7, 9, color='g', alpha=0.2))
axes[1, 0].scatter(np.asarray(perceptron2.xtrain[perct_test_idx1,1:2]),
                np.asarray(perceptron2.xtrain[perct_test_idx1,2:3]),
                facecolors='#4688F1', edgecolors='grey',
                s=60, label='Ground True Class 1 (test)')
axes[1, 0].scatter(np.asarray(perceptron2.xtrain[perct_test_idx2,1:2]),
                np.asarray(perceptron2.xtrain[perct_test_idx2,2:3]),
                marker='+', c='k', s=60, linewidth=2,
                label='Ground True Class -1 (test)')
axes[1, 1].scatter(np.asarray(perceptron2.xtrain[perct_test_idx1a,1:2]),
                np.asarray(perceptron2.xtrain[perct_test_idx1a,2:3]),
                facecolors='#4688F1', edgecolors='grey',
                s=60, label='Estimated Class 1 (test)')
axes[1, 1].scatter(np.asarray(perceptron2.xtrain[perct_test_idx2a,1:2]),
                np.asarray(perceptron2.xtrain[perct_test_idx2a,2:3]),
                marker='+', c='k', s=60, linewidth=2,
                label='Estimated Class -1 (test)')
axes[0,0].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
axes[0,0].legend(frameon=True, fancybox=True)
axes[0,1].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
axes[0,1].legend(frameon=True, fancybox=True)
axes[1,0].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
axes[1,0].legend(frameon=True, fancybox=True)
axes[1,1].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
axes[1,1].legend(frameon=True, fancybox=True)
plt.show()


# check whether standardize data will improve accuracy or not
perceptron3 = Perceptron_binary(X6, Y6, 0.8, randomsplit=True,
                                constant=True, standardized=True)
perceptron3.fitPerceptron(1, 10, 'SGD')
perceptron3.Pct_predict()
print(perceptron3.pct_train_accuracy)
print(perceptron3.pct_test_accuracy)
# the results are quite similar

###############################################################################
# Part III.0 - employ the class to the task III
###############################################################################

class Knn_classify(object):
        """
        This is the class to do the binary classification with Knn algorithm
        The algorithm takes the majority vote rule and does not employ the
        weight for different fold
        As this is nonparametric learnign, there is no need to add the constant
        part into the X
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
                self.X = np.apply_along_axis(Knn_classify.normalize,
                                             axis=0, arr=self.X)
            elif standardized is True:
                self.X = np.apply_along_axis(Knn_classify.standard,
                                             axis=0, arr=self.X)
            if constant is True:
                vector_one = np.ones(self.X.shape[0]).reshape(-1, 1)
                self.X = np.hstack((vector_one, self.X))

            self.xtrain, self.xtest, self.ytrain, self.ytest = (
                Knn_classify.splitSample(self.X, self.Y, percentile, randomsplit))

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

        # def the fitKNN function
        def fitKNN(self, kfold):
            '''
            Input: self.xtrain, self.ytrain, self.xtest, self.ytest
            Defaultly, it employs the euclidean distance to measure
            It assumes that classes are labled as 1 and 2
            Step I: calculate the distance between new x and all the
                    trainning data
            Step II: chose the K nearest neighbors
            Step III: let the neighbors vote
            '''
            # get the key dimensions
            ntrain, mtrain = self.xtrain.shape
            ntest, mtest = self.xtest.shape
            k = kfold
            classIdx = np.zeros([ntest, 1])
            Xindx = np.hstack([self.xtest, classIdx])

            for i in range(ntest):
                xi = Xindx[i, :-1]
                # transer it to matrix format
                xi_mat = np.repeat(xi, ntrain, axis=0)
                xi_diff = np.sqrt(np.sum(np.power(
                    (xi_mat - self.xtrain), 2), axis=1))
                sort_idx = np.argsort(xi_diff, axis=0)
                kfold_idx = sort_idx[0:k]
                class_idt = np.asarray(self.ytrain[kfold_idx, :])
                # majority vote
                class1_no = np.where(class_idt == 1)[0]
                class2_no = np.where(class_idt == 2)[0]
                if len(class1_no) > len(class2_no):
                    Xindx[i, -1] = 1
                else:
                    Xindx[i, -1] = 2

            # return the classifeid vector
            self.Xindx = Xindx[:, -1]

        # calcluate the error
        def predictError(self):
            accuracy = np.sum(self.Xindx == self.ytest, axis=0)/self.ytest.shape[0]
            error = 1 - accuracy
            self.accuracy = accuracy
            self.error = error

###############################################################################
# Part III.1 - test the model, 2 classes and 4 features, kfold = 3
###############################################################################

part7wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part7wine = part7wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part7wine['newclass'] = np.where(part7wine['WineClass'] == 1, 1, 2)
part7wine = part7wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]

X7 = np.asmatrix(part7wine.iloc[:, 1:]).reshape(-1, 4)
Y7 = np.asmatrix(part7wine.newclass).reshape(-1, 1)

knn1 = Knn_classify(X7, Y7, 0.8, randomsplit=False)
knn1.fitKNN(3)
knn1.Xindx
knn1.predictError()
knn1.error

# random split 60%
knn2 = Knn_classify(X7, Y7, 0.6, randomsplit=True)
knn2.fitKNN(3)
knn2.Xindx
knn2.predictError()
knn2.error


###############################################################################
# Part III.2 - test the model, 2 classes and 6 features, kfold = 5
###############################################################################

part8wine = wine.query('WineClass==1| WineClass==2')  # filter 1 and 2 out
part8wine = part8wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity', 'Hue', 'Proline']]
part8wine['newclass'] = np.where(part8wine['WineClass'] == 1, 1, 2)
part8wine = part8wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity', 'Hue', 'Proline']]

X8 = np.asmatrix(part8wine.iloc[:, 1:]).reshape(-1, 6)
Y8 = np.asmatrix(part8wine.newclass).reshape(-1, 1)

knn3 = Knn_classify(X8, Y8, 0.8, randomsplit=True)
knn3.fitKNN(kfold=5)
knn3.Xindx
knn3.predictError()
print(knn3.error)


###############################################################################
# Part III.2 - 2 classes and 6 features, 10-fold cross-validation
###############################################################################

# use X8 and Y8 dataset
# kfold = 3 to 8

kParameter = np.array(range(3, 9))

kfoldcv = 10
test_perform = {}  # initialize a dict to store performance

for lk in kParameter:
    # split the sample
    nrows = X8.shape[0]
    subsetRows = math.floor(nrows/kfoldcv)
    k_randomIndex = random.sample(range(nrows), nrows)

    kfold_perf = {}
    for j in range(kfoldcv):
        subsetIndex = k_randomIndex[j*subsetRows:(j+1)*subsetRows]
        testsubset_x = X8[subsetIndex, :]
        trainsubset_x = np.delete(X8, subsetIndex, 0)
        testsubset_y = Y8[subsetIndex, :]
        trainsubset_y = np.delete(Y8, subsetIndex, 0)
        ntrain, mtrain = trainsubset_x.shape
        ntest, mtest = testsubset_x.shape
        k = lk  # assign the k parameter
        classIdx = np.zeros([ntest, 1])
        Xindx = np.hstack([testsubset_x, classIdx])
        for i in range(ntest):
            xi = Xindx[i, :-1]
            # transer it to matrix format
            xi_mat = np.repeat(xi, ntrain, axis=0)
            xi_diff = np.sqrt(np.sum(np.power(
                (xi_mat - trainsubset_x), 2), axis=1))
            sort_idx = np.argsort(xi_diff, axis=0)
            kfold_idx = sort_idx[0:k]
            class_idt = np.asarray(trainsubset_y[kfold_idx, :])
            # majority vote
            class1_no = np.where(class_idt == 1)[0]
            class2_no = np.where(class_idt == 2)[0]
            if len(class1_no) > len(class2_no):
                Xindx[i, -1] = 1
            else:
                Xindx[i, -1] = 2
        labelclass = Xindx[:, -1]
        accuracy = np.sum(labelclass == testsubset_y, axis=0)/testsubset_y.shape[0]
        error = 1 - accuracy
        kfold_perf[str(j)] = error

    test_perform[str(lk)] = kfold_perf

test_perform

print(pd.DataFrame(test_perform))
print('The regulation number can produce the highest test accuracy is:', pd.DataFrame(test_perform).sum(axis=0)/10)

# for 6 features, it's k=5

# Try with 5

knn4 = Knn_classify(X8, Y8, 0.8, randomsplit=True)
knn4.fitKNN(kfold=5)
knn4.Xindx
knn4.predictError()
print(knn4.error)  # [[0.03846154]]

print('The error is much smaller when k =5')
# therefore, the optimal fold is 5 for binary classification with 6 features.

# End of Code

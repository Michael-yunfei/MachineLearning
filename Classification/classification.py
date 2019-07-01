# Linear Methods for Classification
# @Michael

# This code is to practice the linear methods for classification based on
# the exercise of course by Andrew Ng and ELS by Hasti, etc.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import random
import sklearn.linear_model as skl_lm
from scipy.io import loadmat

# check working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Classification')

# Load the data

admiss = pd.read_csv('ex2data1.csv', names=['Exam1', 'Exam2', 'Admit'])
admiss.head()

# plot the dataset
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(admiss.Exam1[admiss.Admit == 0],
           admiss.Exam2[admiss.Admit == 0],
           facecolors='#FFFD38', edgecolors='grey',
           s=60, label='Not Admitted')
ax.scatter(admiss.Exam1[admiss.Admit == 1],
           admiss.Exam2[admiss.Admit == 1],
           marker='+', c='k', s=60, linewidth=2,
           label='Admitted')
ax.set(xlabel='Exam 1 Score', ylabel='Exame 2 Score')
ax.legend(frameon=True, fancybox=True)
plt.show()

# logistic regression


# def transomfration function
def sigmoid(z):
    gz = 1/(1+np.exp(-z))
    return(gz)


sigmoid = np.vectorize(sigmoid)  # vectorize the function

sigmoid(0)  # array(0.5) test the function
sigmoid(np.array(range(9)))


# define the cost function
def LRcostFunction(theta, x, y):
    '''
    Input: theta is k by 1 vector
           X is the n by k matrix
           Y is the n by 1 vector
    '''
    n = y.shape[0]
    logistic = sigmoid(x @ theta)
    J = -1/n * (np.log(logistic).T @ y + np.log(1 - logistic).T @ (1 - y))

    return(J)


# define the gradient function
def LRgradient(theta, x, y):
    '''
    Input: theta is k by 1 vector
           x is the n by k matrix
           y is the n by 1 vector
    '''
    n = y.shape[0]
    logistic = sigmoid(x @ theta)
    grad = (x.T @ (logistic - y))/n

    return(grad.flatten())  # make it flatten for minimizing function


# test the cost function
nrow = admiss.shape[0]
X = np.stack((np.ones([nrow, 1]),
              np.asarray(admiss.Exam1).reshape(-1, 1),
              np.asarray(admiss.Exam2).reshape(-1, 1)),
             axis=1).reshape(-1, 3)
Y = np.asarray(admiss.Admit).reshape(-1, 1)

initial_theta = np.zeros(X.shape[1]).reshape(-1, 1)

LRcostFunction(initial_theta, X, Y)
LRgradient(initial_theta, X, Y)

# Refleciton on stack
# Once load the dataset, it's better to select variables from csv
# and then convert them into matrix format
# and do the stack part in the final step

res = minimize(LRcostFunction, initial_theta, args=(X, Y.flatten()),
               method='BFGS', jac=LRgradient, options={'maxiter': 400})
res.x  # array([-25.16133284,   0.2062317 ,   0.2014716 ])
res.fun


# Gradient Ascent (bath version)
# be careful, now we are maxmimizing the likelihood function
# through the likelihood function, we have to update the coefficients
# after calculating the gradient.
# Warning: this one nees a very large number of interation

def LGD(x, y, alpha, max):
    '''
    Though the name is called LGD(logistic gradient descent), with the log
    likelihood function, it's more proper to call it logistic gradient descent
    Input:
    x - n by m matrix
    y - n by 1 matrix
    max - maxiterate
    Theta (coefficents) will be generated inside the function

    Output:
    estimated coefficients
    '''
    n, m = x.shape  # get the shape of x: n by m
    theta = np.array([0.0 for i in range(m)]).reshape(-1, 1)  # m by 1 vector
    for i in range(max):
        fx = sigmoid(x @ theta)
        theta = theta - 1/n * alpha * x.T @ (fx-y)

    return(theta)


coef1 = LGD(X, Y, 0.001, 3000000)
coef1
# array([[-20.20395536],
#        [  0.16661406],
#        [  0.1613447 ]])


def LSGD(x, y, alpha, max):
    n, m = x.shape  # get the shape of x: n by m
    theta = np.array([0.0 for i in range(m)]).reshape(-1, 1)  # m by 1 vector
    for i in range(max):
        randomIndex = random.sample(range(n), n)
        xtrain = x[randomIndex]
        ytrain = y[randomIndex]
        for j in range(n):
            x_j = xtrain[j].reshape(1, -1)
            y_j = ytrain[j].reshape(-1, 1)
            fx = sigmoid(x_j @ theta)
            theta = theta - 1/n * alpha * x_j.T @ (fx-y_j)

    return(theta)


ceof2 =LSGD(X, Y, 0.001, 100000)
ceof2
# array([[-4.83184987],
#        [ 0.045346  ],
#        [ 0.03813775]])


# gradient descent methods are not efficient for estimating coefficients of
# logistic regressions. Newton's methods are better


# K-means classification
# Case studies from NG, Adrew

kdata1 = loadmat('ex7data2.mat')
kdata1.keys()
kx1 = kdata1['X']
plt.scatter(kx1[:,0], kx1[:,1], s=40, cmap=plt.cm.prism)

# the plot implies there might be 3 clusters, our taks is to classify them
# the key of doing k-cluster classificiation is to find the centroids

initial_centroids = np.array([[3., 3,], [6., 2.], [8., 5.]])


def Kcentroids(X, initial_cent, stop_rule):
    '''
    The function to find the centroids for k-clusters
    Input:
    X - n by m matrix
    ini_cent: k by m matrix
    stop_rule: the rule for stoping the iteration when convegencing

    output:
    X with classification: b by (m+1) matrix
    kcentroids: k by m matrix for centroids
    distoration function (J) value
    '''
    # get the key dimensions
    int_cent = np.copy(initial_cent)
    n, m = X.shape
    k = int_cent.shape[0]
    # creat the k-cluster index array
    kindx = np.array(range(k))+1
    # initialize the random classification index for X
    int_index = np.random.randint(1, k+1, size=(n, 1))
    # hstack matrix, Xindx is a n by (m+1) matrix with classification index
    Xindx = np.hstack((X, int_index))
    # calculate the initial J value
    Jvalue = 0
    for ki in range(len(kindx)):
        class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
        k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
        Jvalue_k = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
        Jvalue += Jvalue_k

    J_diff = Jvalue-0  # initial difference

    # start to iterate
    while J_diff > stop_rule:
        # calcuate the start J value
            J_startvalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
                Jvalue_k0 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_startvalue += Jvalue_k0

            for ni in range(n):
                x_ni = Xindx[ni, :-1]
                #calculate the difference for each centroids
                x_k_diff_mat = np.asmatrix([x_ni]*k) - int_cent
                x_k_diff_vector = np.sum(np.power(x_k_diff_mat, 2), axis=1)
                find_min = np.where(
                    x_k_diff_vector == np.min(x_k_diff_vector))[0]
                # update the index
                Xindx[ni, -1] = kindx[find_min]

            # update the k-custer centroids
            for kj in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[kj])[0]
                kj_centroid = np.mean(Xindx[class_filter, :-1], axis=0)
                int_cent[kj, :] = kj_centroid

            # calcuate the updated J value
            J_updatevalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
                Jvalue_k1 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_updatevalue += Jvalue_k1

            J_diff = np.abs(J_updatevalue - J_startvalue)

    return (Xindx, int_cent, J_updatevalue)


x_update1, kcent1, jvalue1 = Kcentroids(kx1, initial_centroids, 0.1)

# stop rule = 0.1
kcent1
# array([[1.95399466, 5.02557006],
#        [3.04367119, 1.01541041],
#        [6.03366736, 3.00052511]])
jvalue1
# 266.65851965491936

x_update2, kcent2, jvalue2 = Kcentroids(kx1, initial_centroids, 0.01)
# stop rule = 0.01
kcent2
# array([[1.95399466, 5.02557006],
#        [3.04367119, 1.01541041],
#        [6.03366736, 3.00052511]])
jvalue2
# 266.65851965491936


# add the max iteration rule
def Kcentroids(X, initial_cent, stop_rule, maxiter):
    '''
    The function to find the centroids for k-clusters
    Input:
    X - n by m matrix
    ini_cent: k by m matrix
    stop_rule: the rule for stoping the iteration when convegencing

    output:
    X with classification: b by (m+1) matrix
    kcentroids: k by m matrix for centroids
    distoration function (J) value
    '''
    # get the key dimensions
    int_cent = np.copy(initial_cent)
    n, m = X.shape
    k = int_cent.shape[0]
    # creat the k-cluster index array
    kindx = np.array(range(k))+1
    # initialize the random classification index for X
    int_index = np.random.randint(1, k+1, size=(n, 1))
    # hstack matrix, Xindx is a n by (m+1) matrix with classification index
    Xindx = np.hstack((X, int_index))
    # calculate the initial J value
    Jvalue = 0
    for ki in range(len(kindx)):
        class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
        k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
        Jvalue_k = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
        Jvalue += Jvalue_k

    J_diff = Jvalue-0  # initial difference
    iter = 0

    # start to iterate
    while J_diff > stop_rule and iter < maxiter:
        # calcuate the start J value
            J_startvalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
                Jvalue_k0 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_startvalue += Jvalue_k0

            for ni in range(n):
                x_ni = Xindx[ni, :-1]
                #calculate the difference for each centroids
                x_k_diff_mat = np.asmatrix([x_ni]*k) - int_cent
                x_k_diff_vector = np.sum(np.power(x_k_diff_mat, 2), axis=1)
                find_min = np.where(
                    x_k_diff_vector == np.min(x_k_diff_vector))[0]
                # update the index
                Xindx[ni, -1] = kindx[find_min]

            # update the k-custer centroids
            for kj in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[kj])[0]
                kj_centroid = np.mean(Xindx[class_filter, :-1], axis=0)
                int_cent[kj, :] = kj_centroid

            # calcuate the updated J value
            J_updatevalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.asmatrix([int_cent[ki]]*len(class_filter))
                Jvalue_k1 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_updatevalue += Jvalue_k1

            J_diff = J_updatevalue - J_startvalue
            iter += 1
    final_cent = int_cent

    return (Xindx, final_cent, J_updatevalue)


x_update, kcent, jvalue = Kcentroids(kx1, initial_centroids, 0.1, 2)

kcent
# array([[2.42830111, 3.15792418],
#        [5.81350331, 2.63365645],
#        [7.11938687, 3.6166844 ]])

jvalue
# 1097.776943847284


x_update, kcent, jvalue = Kcentroids(kx1, initial_centroids, 0.01, 100)

kcent
# array([[1.95399466, 5.02557006],
#        [3.04367119, 1.01541041],
#        [6.03366736, 3.00052511]])
jvalue
# 266.65851965491936

# array([[2.42830111, 3.15792418],
#        [5.81350331, 2.63365645],
#        [7.11938687, 3.6166844 ]])


# soft max


###############################################################################
# Bonus - Employ the k-cluster to do binary classification
###############################################################################

class Knn_cluster (object):
    """
    This is the class to do the binary classification with K-nearest-neighbors
    First, it will calcuate the k-centroids;
    After calculating k-centroids, use it to classify the test dataset
    The classification is based on the majority vote rule
    As this is unsurpervised learning, there is no prediction for trainning
    dataset
    # Althoug it keeps the initialize process same, but people should
    # not add constant part
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
            self.X = np.apply_along_axis(Knn_cluster .normalize,
                                         axis=0, arr=self.X)
        elif standardized is True:
            self.X = np.apply_along_axis(Knn_cluster .standard,
                                         axis=0, arr=self.X)
        if constant is True:
            vector_one = np.ones(self.X.shape[0]).reshape(-1, 1)
            self.X = np.hstack((vector_one, self.X))

        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            Knn_cluster .splitSample(self.X, self.Y, percentile, randomsplit))

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

    # def fitKNN
    def fitKNN(self, kfold, maxiter):
        '''
        The function to find the centroids for k-clusters
        Input:
        X - n by m matrix
        kfold: k = 2, as we are doing binary classification
        output:
        X with classification: b by (m+1) matrix
        kcentroids: k by m matrix for centroids
        distoration function (J) value
        '''
        # initialize the centroids
        n, m = self.xtrain.shape
        k = kfold
        int_cent = self.xtrain[0:k, :]
        # creat the k-cluster index array
        kindx = np.array(range(k))+1
        # initialize the random classification index for X
        int_index = np.random.randint(1, k+1, size=(n, 1))
        # hstack matrix, Xindx is a n by (m+1) matrix with classification index
        Xindx = np.hstack((self.xtrain, int_index))
        # calculate the initial J value
        Jvalue = 0
        for ki in range(len(kindx)):
            class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
            k_mat = np.repeat(int_cent[ki, :], len(class_filter), axis=0)
            Jvalue_k = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
            Jvalue += Jvalue_k

        # J_diff = Jvalue-0  # initial difference
        # start to iterate
        # stop when J_diff < 0.001
        iter = 0
        while iter < maxiter:
            # calcuate the start J value
            J_startvalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.repeat(int_cent[ki, :], len(class_filter), axis=0)
                Jvalue_k0 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_startvalue += Jvalue_k0

            for ni in range(n):
                x_ni = Xindx[ni, :-1]
                #calculate the difference for each centroids
                x_k_diff_mat = np.repeat(x_ni, k, axis=0) - int_cent
                x_k_diff_vector = np.sum(np.power(x_k_diff_mat, 2), axis=1)
                find_min = np.where(
                    x_k_diff_vector == np.min(x_k_diff_vector))[0]
                # update the index
                Xindx[ni, -1] = kindx[find_min]

            # update the k-custer centroids
            for kj in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[kj])[0]
                kj_centroid = np.mean(Xindx[class_filter, :-1], axis=0)
                int_cent[kj, :] = kj_centroid

            # calcuate the updated J value
            J_updatevalue = 0
            for ki in range(len(kindx)):
                class_filter = np.where(Xindx[:, -1] == kindx[ki])[0]
                k_mat = np.repeat(int_cent[ki, :], len(class_filter), axis=0)
                Jvalue_k1 = np.sum(np.power(Xindx[class_filter, :-1] - k_mat, 2))
                J_updatevalue += Jvalue_k1

            iter += 1

        self.Xindx = Xindx
        self.Jvalue = J_updatevalue
        self.kcentroids = int_cent


###############################################################################
# Bonus - test the model, 2 classes and 4 features, kfold=2
###############################################################################

# binary with 4 features test
# use the same dataset from Part II
part7wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part7wine = part7wine[['WineClass','Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]
part7wine['newclass'] = np.where(part7wine['WineClass'] == 1, 1, 2)
part7wine = part7wine[['newclass', 'Alcohol','Flavanoids',
                       'Proanthocyanins','Color intensity']]

X7 = np.asmatrix(part7wine.iloc[:, 1:]).reshape(-1, 4)
Y7 = np.asmatrix(part7wine.newclass).reshape(-1, 1)


knn1 = Knn_cluster (X7, Y7, 1, randomsplit=False)
knn1.fitKNN(kfold=2, maxiter=10)
knn1.Jvalue

# Warning:
# as KNN is nonparametric learning, therefore, for the binary class
# with kfold =2,it will just classify the x_i into different clusters.
# The algorithm until now does not use the function to match the classification
# labels.Therefore, at this moment,
# the user for this programm need sort out the lables after
# the classification.
# check the lables
knn1.Xindx[:, -1]
# it looks the first half was classified as class 2
# almost all of the second half was classified into class 1
# we know the orignal dataset label the first half as class 1
# therefore, we need adjust the labels to compare the accuracy

print(np.sum((3-knn1.Xindx[:, -1]) == knn1.ytrain, axis=0)/knn1.Xindx.shape[0])

# it's quite impressive that 8 iteration can lead to 80% accuracy

###############################################################################
# Bonus - test the model, 2 classes and 2 features, kfold = 2
###############################################################################

part8wine = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
part8wine = part8wine[['WineClass', 'Proanthocyanins','Alcalinity of ash']]
part8wine['newclass'] = np.where(part8wine['WineClass'] == 1, 1, 2)
part8wine = part8wine[['newclass', 'Proanthocyanins', 'Alcalinity of ash']]
part8wine.shape
part8wine.head()

X8 = np.asmatrix(part8wine.iloc[:, 1:]).reshape(-1, 2)
Y8 = np.asmatrix(part8wine.newclass).reshape(-1, 1)

knn2 = Knn_cluster (X8, Y8, 1, randomsplit=False)
knn2.fitKNN(2, 10)
knn2.Jvalue

print(np.sum((3 - knn2.Xindx[:, -1]) == Y8, axis=0)/knn2.Xindx.shape[0])

print('Why 2 features can have higher prediction accuracy than 4 features?')
print('The answer can be found in the following graph')

from matplotlib.patches import Ellipse, Circle
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].scatter(part6wine.Proanthocyanins[part8wine.newclass==1],
           part6wine['Alcalinity of ash'][part8wine.newclass==1],
           facecolors='#FFFD38', edgecolors='grey',
           s=60, label='Ground True Class 1')
ax[0].scatter(part6wine.Proanthocyanins[part8wine.newclass==2],
           part6wine['Alcalinity of ash'][part8wine.newclass==2],
           marker='+', c='k', s=60, linewidth=2,
           label='Ground True Class 2')
ax[0].add_artist(Ellipse((1, 23), 1, 9, color='b', alpha=0.2))
ax[0].add_artist(Ellipse((2.2, 17), 1.7, 9, color='g', alpha=0.2))
ax[0].add_artist(Ellipse((2.6, 25), 0.5, 2, color='r', alpha=0.2))
ax[0].text(2.4, 24, 'outlier effect', color='r')
ax[0].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
ax[0].legend(frameon=True, fancybox=True)
ax[1].scatter(np.asarray(knn2.Xindx[:, 0:1][knn2.Xindx[:, -1]==2]),
           np.asarray(knn2.Xindx[:, 1:2][knn2.Xindx[:, -1]==2]),
           facecolors='#FFFD38', edgecolors='grey',
           s=60, label='Predicted Class 1')
ax[1].scatter(np.asarray(knn2.Xindx[:, 0:1][knn2.Xindx[:, -1]==1]),
           np.asarray(knn2.Xindx[:, 1:2][knn2.Xindx[:, -1]==1]),
           marker='+', c='k', s=60, linewidth=2,
           label='Predicted Class 2')
ax[1].add_artist(Ellipse((1, 23), 1, 9, color='b', alpha=0.2))
ax[1].add_artist(Ellipse((2.2, 17), 1.7, 9, color='g', alpha=0.2))
ax[1].set(xlabel='Proanthocyanins', ylabel='Alcalinity of ash')
ax[1].legend(frameon=True, fancybox=True)
plt.show()




# End of Code

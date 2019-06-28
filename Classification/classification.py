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

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
initial_centroids.shape

def Kcentroids(X, int_cent, stop_rule):
    '''
    The function to find the centroids for k-clusters
    Input:
    X - n by m matrix
    ini_cent: k by m matrix
    stop_rule: the rule for stoping the iteration when convegencing

    output:
    X with classification: b by (m+1) matrix
    kcentroids: k by m matrix for centroids
    '''
    n, m = X.shape
    k = int_cent.shape[0]











# End of Code

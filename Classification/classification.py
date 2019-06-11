# Linear Methods for Classification
# @Michael

# This code is to practice the linear methods for classification based on
# the exercise of course by Andrew Ng and ELS by Hasti, etc.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os

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
res.x
res.fun

# End of Code

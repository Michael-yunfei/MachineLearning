# Assignment 6 - Neural Network
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
import scipy.io as spo  # for loading matlab file
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

# check working directory
os.getcwd()
# os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A6')
# os.chdir('C:/Users/User/Desktop/ML/assignments/3')

###############################################################################
# Part 0 - Load the Dataset
###############################################################################

# Read the data out of the mentioned files and create a design matrix X
# in which the rows contain the pixel gray levels of each image. Each row
# should contain 400 values. Create also a vector y containing the labels
# associated to each picture

X = pd.read_csv('digits_data.csv', header=None)
y = pd.read_csv('digits_labels.csv', header=None)

X.shape  # (5000, 400), each row contains 400 values, 5000 totally
y.shape  # (5000, 1), labels for each row
np.unique(y)  # digit 0 is indicated by number 10


# load parameters
params_dict = spo.loadmat("WeightsNN.mat")
type(params_dict)  # dict
Theta1 = params_dict['Theta1']
Theta1.shape  # (25, 401) hiderlayer size M = 25
Theta2 = params_dict['Theta2']
Theta2.shape  # (10, 26) K = 10

params = np.vstack([Theta1.reshape((-1, 1), order='F'),
                    Theta2.reshape((-1, 1), order='F')])
params.shape

# check the matchness by reshape it back
# np.array_equal(np.reshape(params[:25*(400+1)], (25, 401), order='F'), Theta1)
# np.array_equal(np.reshape(params[25*(400+1):], (10, 26), order='F'), Theta2)

###############################################################################
# Preliminaries - write the key functions
###############################################################################
# All the formula employed for functions are from P.395 of the book - The
# Elements of Statistical Learning - by
# Since the instruction asks for a loglikelihood function, the cross-entropy
# error function was used.

# we have two layers: one input layers, and one hiddend layers
# this means that we have to make sure two sets of weights will be updated
# properly - they should satisify the back-propagation equations


# def transomfration function
def sigmoid(z):
    gz = 1/(1+np.exp(-z))
    return(gz)


def nn_twolayer_loglikely(params, p, M, K, X, y, regulation):
    # A function to calcualte the cost and gradiet through back-propagation
    # the formular is taken from the book by HTF;
    # HTF: Hastie, Tibshirani, and Friedman
    # Input:
    # params: a vector of weights for two layers, need to be chopped into two
    #            matrix - one for input layer, one for hider layer
    #                     matrix one- two dimensions: M by (p+1)
    #                                 M: hidden_layer_size
    #                                 (p+1): input_size
    #                     matrix two - two dimensions: K by (M+1)
    #                                  K: output_layer_size
    #                                  (M+1): hidden_layer_size
    # p:input_layer_size,size of row vector or number of coluns = 400 in this case
    # M: hidden_layer_size
    # K: num_labels: number of classes, 10 numbers (digits 0 to 9)
    # X, y,
    # regulation paramter: lambda

    # Output:
    #       return cost J and graident for two coefficents
    #       we need those vectors to do minimization
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    n = X.shape[0]  # number of rows
    theta1 = np.reshape(params[:M*(p + 1)], (M, (p + 1)), order='F')
    theta2 = np.reshape(params[M*(p + 1):], (K, (M+1)), order='F')

    "STEP I - forward pass"
    X = np.hstack([np.ones([n, 1]), X])  # add constant, 5000 by 401;
    Z = X @ theta1.T  # 5000 by 25
    Z1 = np.hstack([np.ones([n, 1]), sigmoid(Z)])  # add constant, 5000 by 26;
    Zout = Z1 @ theta2.T  # 5000 by 10
    Yhat = sigmoid(Zout)

    "STEP II - Calcualte the Cost"
    'It is just the loglikelihood function'
    J = 0  # initialize the Cost
    for j in range(n):
        first_half = np.log(Yhat[j, :]) @ y[j, :].T
        second_half = np.log(1 - Yhat[j, :]) @ (1 - y[j, :]).T
        J += np.sum(-(first_half + second_half))  # make it negative

    J = J/n

    # regulation of the cost function
    # no regulation for the constant part
    J += float(regulation)/(2 * n) *(np.sum(np.power(theta1[:,1:], 2))
                              + np.sum(np.power(theta2[:,1:], 2)))


    "STEP III -  Implement backpropagation to compute the gradients"

    grad_theta1 = np.zeros(theta1.shape)  # initialze the gradiet
    grad_theta2 = np.zeros(theta2.shape)

    for b in range(n):
        x = X[b, :]  # 1 by 401;
        z = Z[b, :]  # 1 by 25;
        z1 = Z1[b, :]  # 1 by 26;
        yhat = Yhat[b, :]  # 1 by 10;
        ytrue = y[b, :]  # 1 by 10;

        ydiff = yhat - ytrue  # 1 by 10;
        zv = np.hstack([np.ones([1, 1]), z])  # 1 by 26;
        # be careful: we need element-wise oepration !!!
        z_gradient = np.multiply(sigmoid(zv), (1 - sigmoid(zv)))  # gradient for z, 1 by 26
        'the following is so called backprogation, see P.396 of HTF'
        zdiff =  np.multiply((ydiff @ theta2), z_gradient)  # 1 by 26

        'update theta'
        grad_theta1 += zdiff[:, 1:].T @ x # 25 by 401
        grad_theta2 += ydiff.T @ z1  # 10 by 26

    grad_theta1 = grad_theta1/n
    grad_theta2 = grad_theta2/n

    grad_theta1[:, 1:] += (theta1[:, 1:] * regulation) / n  # regulation
    grad_theta2[:, 1:] += (theta2[:, 1:] * regulation) / n

    grad = np.vstack([grad_theta1.reshape((-1, 1), order='F'),
                        grad_theta2.reshape((-1, 1), order='F')])

    return(J, grad.flatten())  # make it flatten for minimizing function


# forward pass
def forward_pass(X, theta1, theta2):
    n = X.shape[0]
    X = np.asmatrix(X)
    X = np.hstack([np.ones([n, 1]), X])  # add constant, 5000 by 401;
    Z = X @ theta1.T  # 5000 by 25
    Z1 = np.hstack([np.ones([n, 1]), sigmoid(Z)])  # add constant, 5000 by 26;
    Zout = Z1 @ theta2.T  # 5000 by 10
    Yhat = sigmoid(Zout)

    return Yhat


###############################################################################
# Part I - Classification using provided weigths.
###############################################################################

# Task 1 : Display 100 randomly chosen figures in the data set
# show image
x_mat = np.asmatrix(X)
x_mat.shape
y_vect = np.asmatrix(y)
y_vect.shape

# 10 figures
random_index = np.random.choice(x_mat.shape[0], 10)
plt.imshow(x_mat[random_index, :].reshape(-1, 20).T)
# 100 figures
random_index = np.random.choice(x_mat.shape[0], 100)
plt.imshow(x_mat[random_index, :].reshape(200, 200).T)  # not very clear

# Task 2 : Load up the weights Theta1 and Theta2 from paramsNN.mat,
# use the NN associated to these weights to classify the digits
# in 'digits_data.csv'


# load parameters
params_dict = spo.loadmat("WeightsNN.mat")
type(params_dict)  # dict
Theta1 = params_dict['Theta1']
Theta1.shape  # (25, 401) hiderlayer size M = 25
Theta2 = params_dict['Theta2']
Theta2.shape  # (10, 26) K = 10

# classification
yclass = forward_pass(x_mat, Theta1, Theta2)
y_hatlabel = np.array(np.argmax(yclass, axis=1) + 1)
y_hatlabel.shape


# Create a misclassification matrix whose (i,j)th component
# denotes the percentage of times in which the classifier sends the
# figure with label i to the class j.
num_labels = 10
misclassification_matrix_nn = np.zeros([num_labels, num_labels])

labels = np.asarray(np.arange(1, 11))
for l in labels:
    nc = len(y_hatlabel[np.where(y_vect == l)[0]])
    freq = np.unique(y_hatlabel[np.where(y_vect == l)[0]], return_counts=True)
    for ind, num in enumerate(freq[0]):
        misclassification_matrix_nn[l-1, num-1] = freq[1][ind]/nc*100

print(pd.DataFrame(misclassification_matrix_nn,
                   columns=labels, index=labels))


###############################################################################
# Part II - Classification using provided weigths.
###############################################################################

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
n = y.shape[0]
regulation = 1  # lambda
x_mat = np.asmatrix(X)

# creat the indicator matrix for y
ohe = OneHotEncoder(sparse=False)
y_mat = ohe.fit_transform(y)
y_mat.shape  # (5000, 10)


params = np.vstack([Theta1.reshape((-1, 1), order='F'),
                    Theta2.reshape((-1, 1), order='F')])
params.shape


# test functions
J, grad = nn_twolayer_loglikely(params, input_layer_size,
                                hidden_layer_size, num_labels,
                                x_mat, y_mat, regulation)

# Warning: it did not rewrite the test gradient function from matlab !!!
# the nn_twolayer_loglikely(params, p, M, K, X, y, regulation)
# is tested by checking the accuracy:
# 1 - randomly initialize a parameter array of the size of the full
#     network's parameters
# 2 - minimize the nn_twolayer_loglikely(params, p, M, K, X, y, regulation)
# 3 - check the accuracy
# 4 -gives the misclassification_matrix_nn
# the results shows, the prediction accuray is around 99%


params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1)
                           + num_labels * (hidden_layer_size + 1))
          - 0.5) * 0.25

# minimize the objective function
res = minimize(fun=nn_twolayer_loglikely, x0=params,
               args=(input_layer_size, hidden_layer_size,
                     num_labels, X, y_mat, regulation),
               method='TNC', jac=True, options={'maxiter': 250})
res

res.x
res.x.shape

res_theta1 = np.reshape(res.x[:25*(400+1)], (25, 401), order='F')
res_theta2 = np.reshape(res.x[25*(400+1):], (10, 26), order='F')


yhat = forward_pass(X, res_theta1, res_theta2)
y_pred = np.array(np.argmax(yhat, axis=1) + 1)
y_pred
y.shape


print('the accuracy is' + str(np.sum(np.equal(y_pred, y), axis=0)/n))
# 99.38% accuracy

num_labels = 10
misclassification_matrix_nn = np.zeros([num_labels, num_labels])

labels = np.asarray(np.arange(1, 11))
for l in labels:
    nc = len(y_pred[np.where(y_vect == l)[0]])
    freq = np.unique(y_pred[np.where(y_vect == l)[0]], return_counts=True)
    for ind, num in enumerate(freq[0]):
        misclassification_matrix_nn[l-1, num-1] = freq[1][ind]/nc*100

print(pd.DataFrame(misclassification_matrix_nn,
                   columns=labels, index=labels))

# End of Code

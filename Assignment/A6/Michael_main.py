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

X = pd.read_csv('digits_data.csv')
y = pd.read_csv('digits_labels.csv')

X.shape  # (4999, 400), each row contains 400 values, 4999 totally
y.shape  # (4999, 1), labels for each row

# creat the indicator matrix for y
ohe = OneHotEncoder(sparse=False)
y_mat = ohe.fit_transform(y)
y_mat.shape  # (4999, 10)

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

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
n = length(y)
regulation = 0  # lambda

###############################################################################
# Preliminaries - write the key functions
###############################################################################
# All the formula employed for functions are from P.395 of the book - The
# Elements of Statistical Learning - by Hastie, Tibshirani, and Friedman
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
    n = X.shape[0]  # number of rows
    theta1 = np.reshape(params[:M*(p + 1)], (M, (p + 1)), order='F')
    theta2 = np.reshape(params[M*(p + 1):], (K, (M+1)), order='F')
    "Step I - forward pass"
    X = np.hstack([np.ones([n, 1]), X])  # add constant












































# End of Code

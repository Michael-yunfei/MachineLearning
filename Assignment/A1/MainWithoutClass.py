# Assignment 1a, programming without using classes
# Machine Learning
# @ Michael

import numpy as np
import pandas as pd
import scipy.io as spo
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys

# get and set the working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A1')

# load the dataset
# Complementary Information:
# M&Ms dataset: a small dataset about the number of colored M&Ms in packets
# and the packets weight
# (https://gist.github.com/giob1994/ffcd8c72a8a5477219aca9c5884c2094)
# variables
# determine the names of the variables in the dataset with the command
# "who"

MandMs = spo.loadmat("MandMs.mat")
type(MandMs)  # dict
MandMs.keys()
MandMs["Green"]

# transform it into DataFrame format
MandMs = {k: v for k, v in MandMs.items() if k[0] != '_'}  # k-keys; v-values
MandMs = pd.DataFrame({k: [vv[0] for vv in v] for k, v in MandMs.items()})
MandMs.columns  # equivalnent to `who` function in matlab
MandMs.shape

# names of variables
# Index(['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Brown', 'Weight'])

# Explore the dataset
MandMs.head()

# plot all fitted line before doing regressoion
fig, axes = plt.subplots(2, 3, figsize=(12, 9))
i = 0
for m, n in enumerate(MandMs.columns[:6]):
    if m >= 3:
        i = 1
        m -= 3
    else:
        i = 0
    sns.regplot(MandMs[n], MandMs["Weight"], color=n, ax=axes[i, m])


# Write a function to split the sample
def splitSample(sample, trainSize, permute=False):
    """
    Split the tranning sample randomly or percentilely
    Input: sample - the n by m dimension dataset, taken as matrix
           trainSize - the percentage of sample
           permute - boolean value to indicate the random seletion or not
    Output: the selected the trainning dataset and testsample
    """
    try:
        sample = np.asmatrix(sample)
    except Exception:
        sys.exit(0)
        print("make sure you data can be transformed into matrix")

    sample_length = int(sample.shape[0] * trainSize)
    if permute is True:
        random_index = random.sample(range(sample.shape[0]), sample_length)
        trainSample = sample[random_index, :]
        testSample = np.delete(sample, random_index, 0)

        return(trainSample, testSample)
    else:
        percentile_index = list(range(sample_length))
        trainSample = sample[percentile_index, :]
        testSample = np.delete(sample, percentile_index, 0)

        return(trainSample, testSample)


MandMs_matrix = np.asmatrix(MandMs)
trainSample, testSample = splitSample(MandMs_matrix, 0.8, permute=True)
trainSample.shape
testSample.shape


# define the L1 loss function
def AbsoluteLoss(x, y, theta):
    """
    A absolute error loss function to compute the loss (cost) error
    Input: x - matrix n by m
           y - vector n by 1
           theta - vector m(or <=m) by 1, order matters considering the
           intercept
    Output: the sum of squared error
    """
    try:
        x = np.asmatrix(x)
        y = np.asmatrix(y).reshape(-1, 1)
        theta = np.asmatrix(theta).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            x = x.transpose()
    except Exception:
        print('There is an error with the input data,\
              please make sure your x can be transformed into n by m matrix,\
              your y can be transformed into n by 1 vector,\
              your theta can be transformed into m by 1 vector')
        sys.exit(0)

    n = x.shape[0]  # sample size
    fx = x @ theta  # matrix (dot) production for estimated y
    loss = 1/2 * 1/n * np.sum(np.abs(fx - y))

    return(loss)


# define the L2 loss function
def SquareLoss(x, y, theta):
    """
    A square error loss function to compute the loss (cost) error
    Input: x - matrix n by m
           y - vector n by 1
           theta - vector m by 1, order matters considering the intercept
    Output: the sum of squared error
    """
    try:
        x = np.asmatrix(x)
        y = np.asmatrix(y).reshape(-1, 1)
        theta = np.asmatrix(theta).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            x = x.transpose()
    except Exception:
        print('There is an error with the input data,\
              please make sure your x can be transformed into n by m matrix,\
              your y can be transformed into n by 1 vector,\
              your theta can be transformed into m by 1 vector')
        sys.exit(0)

    n = x.shape[0]  # sample size
    fx = x @ theta  # matrix (dot) production for estimated y
    loss = 1/2 * 1/n * np.sum(np.square(fx - y))  # use average with 1/n

    return(loss)


# define the Huber Loss function
def HuberLoss(x, y, theta, epsilon):
    """
    A absolute error loss function to compute the loss (cost) error
    Input: x - matrix n by m
           y - vector n by 1
           theta - vector m(or <=m) by 1, order matters considering
           the intercept
    Output: the sum of squared error
    """
    try:
        x = np.asmatrix(x)
        y = np.asmatrix(y).reshape(-1, 1)
        theta = np.asmatrix(theta).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            x = x.transpose()
    except Exception:
        print('There is an error with the input data,\
              please make sure your x can be transformed into n by m matrix,\
              your y can be transformed into n by 1 vector,\
              your theta can be transformed into m by 1 vector')
        sys.exit(0)

    n = x.shape[0]  # sample size
    fx = x @ theta  # matrix (dot) production for estimated y
    error = np.abs(y - fx)

    def hl(element):
        if element <= epsilon:
            loss = 1/2 * element**2
        else:
            loss = epsilon * element - 1/2 * epsilon**2

        return(loss)

    hlvector = np.vectorize(hl)

    loss = 1/n * np.sum(hlvector(error))

    return(loss)


# Run the regression with Gradient Descent

def GradientDescent(x, y, theta, alpha, tolerate, maxiterate, epsilon):
    """
    Gradient Descent for estimating regression model
    Input: x - matrix n by m
           y - vector n by 1
           theta - vector m(or <=m) by 1, order matters considering
           the intercept
    Output: the sum of squared error
    """
    i = 0  # set the iteration counting index
    tolerate_rule = 1  # set the initial tolerate rate
    n = x.shape[0]
    current_theta = theta
    scost_vector = np.empty([0, 1])
    lcost_vector = np.empty([0, 1])
    hcost_vector = np.empty([0, 1])

    # iterate
    while tolerate_rule >= tolerate and i <= maxiterate:
        sl = np.array(SquareLoss(x, y, current_theta)).reshape([1, 1])
        scost_vector = np.append(scost_vector, sl, axis=0)  # store cost
        abs = np.array(AbsoluteLoss(x, y, current_theta)).reshape([1, 1])
        lcost_vector = np.append(lcost_vector, abs, axis=0)  # store cost
        hbs = np.array(HuberLoss(x, y, current_theta, epsilon)).reshape([1, 1])
        hcost_vector = np.append(hcost_vector, hbs, axis=0)  # store cost
        fx = x @ current_theta
        update_theta = current_theta - alpha * (1/n) * x.transpose() @ (fx - y)
        tolerate_rule = np.min(np.abs(update_theta - current_theta))
        i += 1
        current_theta = update_theta

    cost_matrix = np.asmatrix(np.stack((scost_vector, lcost_vector,
                                        hcost_vector),
                                       axis=-1))
    cost_dataframe = pd.DataFrame(cost_matrix, columns=['AbsoluteLoss',
                                                        'SquareLoss',
                                                        'HuberLoss'])
    return(current_theta, cost_dataframe)


# Construct a linear regression model that predicts the weight of a given bag
# provided the number of red candies in a bag.
# Use 80% of the sample length for training and 20% for testing
# Compute the training (in-sample) and testing (out-of-sample) performance
# of such model using l1, l2 (MSE) and Huber losses.
# Show some usefull figures/charts/plots as stated in the assignment

# split the sample
trainSample, testSample = splitSample(MandMs, 0.8, permute=True)
trainSample = pd.DataFrame(trainSample, columns=MandMs.columns)
testSample = pd.DataFrame(testSample, columns=MandMs.columns)

# fit the regression
input_x = np.asarray(trainSample.Red).reshape(-1, 1)
input_y = np.array(trainSample.Weight).reshape(-1, 1)
alpha = 0.001  # learning rate
theta_initial = np.asmatrix([0])
tolerate = 0.00001
maxiter = 1500
epsilon = 1.13

Est_coefficients, loss_matrix = GradientDescent(input_x, input_y,
                                                theta_initial,
                                                alpha, tolerate, maxiter,
                                                epsilon)

Est_coefficients

# at this stage, I realize it's better to write a class
# End of Code

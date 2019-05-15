# Assignment 1a, programming with using classes
# Machine Learning
# @ Michael, Lada, Coco

import numpy as np
import pandas as pd
import scipy.io as spo
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# get and set the working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A1')


###############################################################################
# Section I - Load the Dataset
###############################################################################
MandMs = spo.loadmat("MandMs.mat")
type(MandMs)  # dict
MandMs.keys()

# transform it into DataFrame format
MandMs = {k: v for k, v in MandMs.items() if k[0] != '_'}  # k-keys; v-values
MandMs = pd.DataFrame({k: [vv[0] for vv in v] for k, v in MandMs.items()})
MandMs.columns  # equivalnent to `who` function in matlab
MandMs.shape

# names of variables
# Index(['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Brown', 'Weight'])

###############################################################################
# Section II - Explore the Dataset
###############################################################################
MandMs.head()
MandMs.describe()

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


###############################################################################
# Section III - Regression
###############################################################################
# use standard gradident descent to estiamte coefficients
# Employ the classes of Python to do the task


class ML_gradient(object):
    """
    A class to do machine learning tast with gradient descent.

    Input:
        X - n by m matrix or dataframe
        Y - n by 1 matrix or dataframe
        boolean:split or not - split gives the train and test sample
        splitratio: get the train sample size

    Main functions:
        split function: split sample into train and test
        Three loss functions: absolute loss, square loss, huber loss
        Gradient_descent function: estimate the coefficients
        Table function: gives the summary of task
        Plot function: gives the plot of regression

    Output:
        different functions gives different returns
    """

    def __init__(self, X, Y, theta, percentile, randomsplit=False):
        '''
        Initilize the input: (order matters)
        Taken X and Y as dataframe or matrix format, but still trying to
        convert to array and matrix
        Default model includes interecpet
        '''
        self.X = X
        self.Y = Y
        self.theta = theta
        try:
            self.X = np.asarray(self.X)
            self.Y = np.asarray(self.Y).reshape(-1, 1)
            self.theta = np.asarray(self.theta).reshape(-1, 1)
            if (self.X.shape[0] != self.Y.shape[0]):
                print('Input Y and X \
                      have different sample size')
                sys.exit(0)
        except Exception:
            print('There is an error with the input data.\
                  Make sure input are either matrix or dataframe')
            sys.exit(0)

        self.xtrain, self.xtest = ML_gradient.splitSample(self.X,
                                                          percentile,
                                                          randomsplit)
        self.ytrain, self.ytest = ML_gradient.splitSample(self.Y,
                                                          percentile,
                                                          randomsplit)

    @staticmethod
    def splitSample(sample, trainSize, permute=False):
        '''
        static function to split the sample
        '''
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

    @staticmethod
    def AbsoluteLoss(x, y, theta):
        n = x.shape[0]  # sample size
        fx = x @ theta  # matrix (dot) production for estimated y
        loss = 1/2 * 1/n * np.sum(np.abs(fx - y))

        return(loss)

    @staticmethod
    def SquareLoss(x, y, theta):
        n = x.shape[0]  # sample size
        fx = x @ theta  # matrix (dot) production for estimated y
        loss = 1/2 * 1/n * np.sum(np.square(fx - y))  # use average with 1/n

        return(loss)

    @staticmethod
    def HuberLoss(x, y, theta, epsilon=1):
        n = x.shape[0]  # sample size
        fx = x @ theta  # matrix (dot) production for estimated y
        error = np.abs(y - fx)

        def hl(element):
            '''
            take element as absolute error
            '''
            if element <= epsilon:
                loss = 1/2 * element**2
            else:
                loss = epsilon * (element - 1/2 * epsilon)

            return(loss)

        hlvector = np.vectorize(hl)

        loss = 1/n * np.sum(hlvector(error))

        return(loss)

    def estimate(self, alpha, tolerate, maxiterate):
        i = 0  # set the iteration counting index
        tolerate_rule = 1  # set the initial tolerate rate
        n = self.xtrain.shape[0]
        current_theta = self.theta
        cost_vector = np.empty([0, 1])
        theta_matrix = np.empty([0, self.theta.shape[0]])

        # iterate
        while tolerate_rule >= tolerate and i <= maxiterate:
            sl = np.array(ML_gradient.SquareLoss(
                self.xtrain, self.ytrain, current_theta)).reshape([1, 1])
            cost_vector = np.append(cost_vector, sl, axis=0)
            theta_matrix = np.append(theta_matrix,
                                     current_theta.reshape(1, -1),
                                     axis=0)
            fx = self.xtrain @ current_theta
            update_theta = (current_theta
                            - alpha * (1/n) *
                            self.xtrain.transpose() @ (fx - self.ytrain))
            tolerate_rule = np.max(np.abs(update_theta
                                          - current_theta))
            i += 1
            current_theta = update_theta

        self.coeff = current_theta
        return(current_theta, cost_vector, theta_matrix)

    def performance(self):
        """
        Insample performace with three loss functions
        Outsample performace with three loss functions
        """
        insample_aberror = ML_gradient.AbsoluteLoss(self.xtrain,
                                                    self.ytrain,
                                                    self.coeff.reshape(-1, 1))
        insample_sqerror = ML_gradient.SquareLoss(self.xtrain,
                                                  self.ytrain,
                                                  self.coeff.reshape(-1, 1))
        insample_hberror = ML_gradient.HuberLoss(self.xtrain,
                                                 self.ytrain,
                                                 self.coeff.reshape(-1, 1))
        insample_per = np.array([insample_aberror,
                                 insample_sqerror,
                                 insample_hberror]).reshape(-1, 1)

        outsample_aberror = ML_gradient.AbsoluteLoss(
            self.xtest, self.ytest, self.coeff.reshape(-1, 1))
        outsample_sqerror = ML_gradient.SquareLoss(
            self.xtest, self.ytest, self.coeff.reshape(-1, 1))
        outsample_hberror = ML_gradient.HuberLoss(
            self.xtest, self.ytest, self.coeff.reshape(-1, 1))
        outsample_per = np.array([outsample_aberror,
                                 outsample_sqerror,
                                 outsample_hberror]).reshape(-1, 1)

        per_dataframe = pd.DataFrame(np.hstack([insample_per,
                                                outsample_per]),
                                     columns=['Insample Loss',
                                              'Outsample Loss'],
                                     index=['Absolute',
                                            'Square',
                                            'Huber'])
        return(per_dataframe)

    def trainplot2D(self, xlab):
        '''
        Predicted weight values (regression line), predicted data points
        and ground-truth data points for the training data.
        '''
        xdomain = np.linspace(0, np.max(self.xtrain))
        yfitline = self.coeff[0] + self.coeff[1] * xdomain
        fig, ax = plt.subplots(figsize=(6, 5), sharex=True)
        ax.plot(self.xtrain, self.ytrain, 'o', color='#4688F1')
        ax.plot(xdomain, yfit, color='#F67770')
        ax.set(xlabel=xlab, ylabel='Weight',
               title='Ground-truth data, predicted data,\
               and fitted regression line')








# initialize the class
# make insure input are in matrix format
input_x =  np.hstack([np.ones(MandMs.shape[0]).reshape(-1, 1),
                      np.asarray(MandMs.Red).reshape(-1, 1)])
input_y = np.asmatrix(MandMs['Weight']).reshape(-1, 1)
theta_initial = [0, 0]

gradientEs = ML_gradient(input_x, input_y, theta_initial, 0.8)
a, b, c = gradientEs.estimate(0.01, 0.00001, 10000)
a  # array([[4.29550515]])
gradientEs.performance()


np.max(MandMs.Red)

xdomain = np.linspace(0, 22)
yfit = a[0] + a[1] * xdomain

fig, ax = plt.subplots(figsize=(6, 5), sharex=True)
ax.plot(MandMs.Red, input_y, 'o', color='#4688F1')
ax.plot(xdomain, yfit, color='#F67770')
fig.show()

abc = np.array()





input_y
checkmodel = LinearRegression()
checkmodel.fit(input_x, input_y)
checkmodel.intercept_
checkmodel.coef_

checkstmodel = sm.OLS(input_y, input_x).fit()
checkstmodel.summary()









# End of Code

# Assignment 2, programming with using classes
# Machine Learning
# @ Coco, Lada, Michael

import numpy as np
import pandas as pd
import scipy.io as spo
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import statsmodels.api as sm

# get and set the working directory
# please set your own working directory which inlcudes the dataset
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A2')

###############################################################################
# Section I - Load the Dataset
###############################################################################

whr = pd.read_csv('WHR2016.csv')
whr.head()
whr.columns
whr.describe()

# plot the correlation heatmap
corr = whr.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()
# we can see that happiness score has high collelation with most economic
# variables and freedom. This gives the premiliary ideas on regression

###############################################################################
# Section III - Regression
###############################################################################


# Employ the classes of Python to do the task
###############################################################################

class LP_regression(object):
    """
    A class to do machine learning tast with:
        OSL
        Bath Gradient Method
        Stochastic Gradient Method

    Input:
        X - n by m matrix or dataframe
        Y - n by 1 matrix or dataframe
        boolean:split or not - split gives the train and test sample
        splitratio: get the train sample size
        Polynomial degree: k, e.g. k = 10 <=> x^10

    Main functions:
        OLS function: fit regression with norm
        square loss: square loss function
        BGD function: bath gradient descent method
        SGD function: stochasitc gradient descent method
        Table function: gives the summary of task

    Output:
        different functions gives different returns
    """

    def __init__(self, X, Y, percentile, randomsplit=False):
        '''
        Initilize the input: (order matters)
        Taken X and Y as dataframe or matrix format, but still trying to
        convert to array and matrix
        Default model includes interecpet
        percential: the ratio for splitting the sample
        randomsplit: if it is true, sample is splited  randomly
        '''
        self.X = X
        self.Y = Y
        self.percentile = percentile
        try:
            self.X = np.asarray(self.X)
            self.Y = np.asarray(self.Y).reshape(-1, 1)
            if (self.X.shape[0] != self.Y.shape[0]):
                print('Input Y and X \
                      have different sample size')
                sys.exit(0)
        except Exception:
            print('There is an error with the input data.\
                  Make sure input are either matrix or dataframe')
            sys.exit(0)

        self.xtrain, self.xtest, self.ytrain, self.ytest = (
            LP_regression.splitSample(self.X, self.Y, percentile, randomsplit))

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

    def OLS(self):
        '''
        Estimate the coefficients, give
        with normalize X or standardize X
        '''

        # use betahat = inv(X'X)X'Y to get the results
        self.olsBetahat = (
            np.linalg.inv(
                np.transpose(
                    self.xtrain)@self.xtrain)@np.transpose(
                        self.xtrain)@self.ytrain)
        # estimation for beta
        # Calculate the standard errors
        self.olsYhat = self.xtrain @ self.olsBetahat
        self.olsResidu = self.ytrain - self.olsYhat

    def BGD(self, theta, alpha, tolerate, maxiterate):
        '''
        Bath Gradient Descent method to estimate coefficients
        '''
        i = 0  # set the iteration counting index
        tolerate_rule = 1  # set the initial tolerate rate
        n = self.xtrain.shape[0]
        current_theta = np.asarray(theta).reshape(-1, 1)

        # iterate
        while tolerate_rule >= tolerate and i <= maxiterate:
            fx = self.xtrain @ current_theta
            update_theta = (current_theta
                            - alpha * (1/n)
                            * self.xtrain.transpose() @ (fx - self.ytrain))
            tolerate_rule = np.max(np.abs(update_theta
                                          - current_theta))
            i += 1
            current_theta = update_theta

        self.bgdBetahat = current_theta
        return(current_theta)

    def SGD(self, theta, maxiterate):
        '''
        Stochastic gradient descent method
        input: initial value for theta
               maxiterate to do iteration
        '''
        n = self.xtrain.shape[0]
        current_theta2 = np.asarray(theta).reshape(-1, 1)
        alpha0, alpha1 = 5, 50
        for it in range(maxiterate):
            randomIndex = random.sample(range(n), n)
            xtrain = self.xtrain[randomIndex]
            ytrain = self.ytrain[randomIndex]
            for i in range(n):
                x_i = xtrain[i].reshape(1, -1)
                y_i = ytrain[i].reshape(-1, 1)
                alpha = alpha0/(alpha1 + it*n+i)
                fx = x_i @ current_theta2
                update_theta2 = (current_theta2
                                 - alpha * 2
                                 * x_i.transpose() @ (fx - y_i))
                current_theta2 = update_theta2

        self.sgdBetahat = current_theta2
        self.sgdYhat = self.xtrain @ self.sgdBetahat
        return(current_theta2)

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

    def performance(self, method):
        """
        Insample performace with three loss functions
        Outsample performace with three loss functions
        """
        if method == 'OLS':
            self.coeff = self.olsBetahat
        elif method == 'BGD':
            self.coeff = self.bgdBetahat
        elif method == 'SGD':
            self.coeff = self.sgdBetahat
        else:
            print("Make sure your method is one of those:\
                  'OLS', 'BGD', 'SGD'")
        if self.percentile == 1:
            insample_aberror = LP_regression.AbsoluteLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_sqerror = LP_regression.SquareLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_hberror = LP_regression.HuberLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_per = np.array([insample_aberror,
                                     insample_sqerror,
                                     insample_hberror]).reshape(-1, 1)
            per_dataframe = pd.DataFrame(insample_per,
                                         columns=['Fullsample Loss'],
                                         index=['Absolute',
                                                'Square',
                                                'Huber'])
        else:
            insample_aberror = LP_regression.AbsoluteLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_sqerror = LP_regression.SquareLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_hberror = LP_regression.HuberLoss(
                self.xtrain, self.ytrain, self.coeff.reshape(-1, 1))
            insample_per = np.array([insample_aberror,
                                     insample_sqerror,
                                     insample_hberror]).reshape(-1, 1)

            outsample_aberror = LP_regression.AbsoluteLoss(
                self.xtest, self.ytest, self.coeff.reshape(-1, 1))
            outsample_sqerror = LP_regression.SquareLoss(
                self.xtest, self.ytest, self.coeff.reshape(-1, 1))
            outsample_hberror = LP_regression.HuberLoss(
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


###############################################################################
# Single variable regression
###############################################################################

# scatterplot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(whr.Freedom, whr["Happiness Score"])
ax.set(xlabel='Freedom', ylabel='Happiness Score',
       title='Scatter plot of main variables')
plt.show()

# initialize the class to fit the regression
# make insure input are in matrix format and add constant
input_x = np.hstack([np.ones(whr.shape[0]).reshape(-1, 1),
                     np.asarray(whr.Freedom).reshape(-1, 1)])

input_y = np.asmatrix(whr["Happiness Score"]).reshape(-1, 1)

task1 = LP_regression(input_x, input_y, 1)
task1.OLS()  # estimate coefficients with OLS
task1.olsBetahat  # array([[3.73221579],[4.44742889]])
print(task1.performance(method='OLS'))  # different loss


# check results with python package
model1 = sm.OLS(task1.ytrain, task1.xtrain).fit()
model1.summary()  # it's same value in 4 digits

# scatter plot with the fitted line
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(whr.Freedom, whr["Happiness Score"])
ax.plot(whr.Freedom, task1.olsYhat, color='#F34235')
ax.set(xlabel='Freedom', ylabel='Happiness Score',
       title='Scatter plot of main variables')
plt.show()

# estimate coefficients with batch gradient descent
theta_initial1 = [0, 0]  # set initial value
alpha1 = 0.01  # set learning rate, don't set it > 0.01
tolerate1 = 0.00000001
maxiter1 = 15000

task1.BGD(theta_initial1, alpha1, tolerate1, maxiter1)
task1.bgdBetahat  # array([[3.79527656], [4.2805871 ]])

# estimate coefficients with stochastic gradiet descent

task1.SGD(theta_initial1, 100)
task1.sgdBetahat  # array([[4.04331295], [3.6267929 ]])

# until now, it works not that well for SGD

###############################################################################
# Polynomial Regression with One Regressor:
###############################################################################
# k = 10
# x = freedome score

# Prepare the dataset
k = 10
input_y = np.asmatrix(whr["Happiness Score"]).reshape(-1, 1)
# constrcut a matrix to store all X (x^0 to x^10)
poly_Xmatrix = np.ones(whr.shape[0]).reshape(-1, 1)
for i in range(k):
    poly_Xmatrix = np.hstack([poly_Xmatrix,
                              np.asarray(whr.Freedom).reshape(-1, 1)**(i+1)])

# OLS
ols_poly_coeffs = {}
ols_poly_sqloss = {}
# initialize two dictionary to store all performace
# constrcut a matrix to store all X (x^0 to x^10)

fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
fig.suptitle(
    'All fitted line with polynomial regressions (k up to 10)')
for i in range(k):
    input_xi = poly_Xmatrix[:, 0:(i+2)]
    modename = 'model'+str(i+1)
    olsreg = LP_regression(input_xi, input_y, 1)
    olsreg.OLS()  # estimate coefficients with OLS
    ols_poly_coeffs[modename] = olsreg.olsBetahat
    ols_poly_sqloss[modename] = olsreg.performance(method='OLS').values[1]
    ax = fig.add_subplot(3, 4, i+1)
    ax.scatter(olsreg.xtrain[:, 1], olsreg.ytrain)
    ax.scatter(olsreg.xtrain[:, 1], olsreg.olsYhat, color='#F67770')
    ax.set(xlabel='Freedom', ylabel='Happiness Score')
    ax = {}
plt.show()

print(ols_poly_coeffs)
print(pd.DataFrame(ols_poly_sqloss))

# SGD
sgd_poly_coeffs = {}
sgd_poly_sqloss = {}
# initialize two dictionary to store all performace

fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
fig.suptitle(
    'All fitted line with polynomial regressions (k up to 10)')
for i in range(k):
    input_xi = poly_Xmatrix[:, 0:(i+2)]
    modename = 'model'+str(i+1)
    sgdreg = LP_regression(input_xi, input_y, 1)  # initialize class
    theta = np.random.randn(input_xi.shape[1], 1)
    sgdreg.SGD(theta, 100)  # estimate coefficients with OLS
    sgd_poly_coeffs[modename] = sgdreg.sgdBetahat
    sgd_poly_sqloss[modename] = sgdreg.performance(method='SGD').values[1]
    ax = fig.add_subplot(3, 4, i+1)
    ax.scatter(sgdreg.xtrain[:, 1], sgdreg.ytrain)
    ax.scatter(sgdreg.xtrain[:, 1], sgdreg.sgdYhat, color='#F67770',
               label='Fitted line')
    ax.set(xlabel='Freedom', ylabel='Happiness Score')
    ax = {}
plt.show()

print(sgd_poly_coeffs)
print(pd.DataFrame(sgd_poly_sqloss))

###############################################################################
# Regression with Two Regressors:
###############################################################################










# End of Code

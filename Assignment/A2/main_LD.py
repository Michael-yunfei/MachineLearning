# Machine Learning
# Assignment 2
# Wenxuan Zhang  01/945008
# Fei Wang 01/942870
# Lada Rudnitckaia 01/942458



#  The basic structure of the exercise is taken from Andrew Ng

#  Linear and polynomial regression with multiple variables

#  Instructions
#  ------------

#  This file contains code that helps you to get started on the
#  polynomial regression exercise.

#  You will need to complete the following functions in this
#  exericse:

#     gradientDescent.m
#     computeLossLinModel.m
#     computeLoss.m
#     gradientDescentStochastic.m
#     featureNormalize.m
#     normalEqn.m


#######          Due to : 31.05.2019 23:59

## Part I: Load Data ======================================================
# Load All Data discarding the first three columns containing Country,
# Region, HappinessRank
import numpy as np
import pandas as pd
import scipy.io as spo
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import statsmodels.api as sm
from mpl_toolkits import mplot3d
import time
import math
from sklearn.linear_model import LinearRegression

# please set your own working directory which inlcudes the dataset
os.getcwd()
os.chdir('C:/Users/User/Documents/GitHub/assignment-2-rudnitckaiawangzhang')

whr = pd.read_csv('WHR2016.csv')
whr.head()
whr.columns
whr.describe()

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

    def KFoldXV(self, kfold, method):
        '''
        A k-fold cross-validation function
        Input: k-fold, such as 5 fold
               method: BGD, SGD, or OLS
        output: the k-fold losses stored in a dictionary
                the k-fold coefficients stored in a dictionary
        '''
        # split the dataset index
        nrows = self.X.shape[0]
        ncolumns = self.X.shape[1]
        subsetRows = math.floor(nrows/kfold)
        randomIndex = random.sample(range(nrows), nrows)
        crossCoeff = {}  # initialize a dict to store coefficients
        crossPerfom = {}  # initialize a dict to store performance
        for i in range(kfold):
            subsetIndex = randomIndex[i*subsetRows:(i+1)*subsetRows]
            testsubset_x = self.X[subsetIndex, :]
            trainsubset_x = np.delete(self.X, subsetIndex, 0)
            testsubset_y = self.Y[subsetIndex, :]
            trainsubset_y = np.delete(self.Y, subsetIndex, 0)
            if method == 'OLS':
                olsMethod = LP_regression(trainsubset_x, trainsubset_y, 1)
                olsMethod.OLS()
                modename = 'OLSFold'+str(i+1)
                crossCoeff[modename] = olsMethod.olsBetahat
                olsLoss = LP_regression.SquareLoss(testsubset_x,
                                                   testsubset_y,
                                                   olsMethod.olsBetahat)
                crossPerfom[modename] = olsLoss
            elif method == 'BGD':
                bgdMethod = LP_regression(trainsubset_x, trainsubset_y, 1)
                thetaInitial = np.random.rand(ncolumns, 1)
                bgdMethod.BGD(thetaInitial, 0.01, 0.00001, 15000)
                modename = 'BGDFold'+str(i+1)
                crossCoeff[modename] = bgdMethod.bgdBetahat
                bgdLoss = LP_regression.SquareLoss(testsubset_x,
                                                   testsubset_y,
                                                   bgdMethod.bgdBetahat)
                crossPerfom[modename] = bgdLoss
            elif method == 'SGD':
                sgdMethod = LP_regression(trainsubset_x, trainsubset_y, 1)
                thetaInitial = np.random.rand(ncolumns, 1)
                sgdMethod.SGD(thetaInitial, 200)
                modename = 'SGDFold'+str(i+1)
                crossCoeff[modename] = sgdMethod.sgdBetahat
                sgdLoss = LP_regression.SquareLoss(testsubset_x,
                                                   testsubset_y,
                                                   sgdMethod.sgdBetahat)
                crossPerfom[modename] = sgdLoss
            else:
                print("Make sure your method is one of those:\
                      'OLS', 'BGD', 'SGD'")
                sys.exit(0)

        return(crossCoeff, crossPerfom)

    def predict(self, method):
        if method == 'BGD':
            coeff = self.bgdBetahat
        elif method == "SGD":
            coeff = self.sgdBetahat
        else:
            coeff = self.olsBetahat
        insample_ypredic = self.xtrain @ coeff
        outsample_ypredic = self.xtest @ coeff

        return(insample_ypredic, outsample_ypredic)


## Part II: Regression with one regressor =================================

###########################################################################
# Task 1 : Create a regressor (input, feature) X using the "Freedom" score and
# an explained variable (output, response) Y from "HappinessScore".
# Plot the scatter plot of both variables. Use all available countries.
###########################################################################
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(whr.Freedom, whr["Happiness Score"])
ax.set(xlabel='Freedom', ylabel='Happiness Score',
       title='Task 1, Scatter plot of main variables')
plt.show()

###########################################################################
# Task 2 : Train a linear regression model (see Assignment 1) using the
# whole sample and the created regressor X. In order to estimate parameters
# beta, use first the normal equations.
###########################################################################
input_x = np.hstack([np.ones(whr.shape[0]).reshape(-1, 1),
                     np.asarray(whr.Freedom).reshape(-1, 1)])
input_y = np.asmatrix(whr["Happiness Score"]).reshape(-1, 1)

task1 = LP_regression(input_x, input_y, 1)
task1.OLS()  # estimate coefficients with OLS
task1.olsBetahat  # array([[3.73221579],[4.44742889]])
print('Task 2, coefficients using normal equation:\n', task1.olsBetahat)
print('Task 2, losses using normal equation:\n', task1.performance(method='OLS'))

#           Fullsample Loss
# Absolute         0.390514
# Square           0.439503
# Huber            0.397632

###########################################################################
# Task 3 : Check the computations with built-in regress function.
model1 = sm.OLS(task1.ytrain, task1.xtrain).fit()
model1.summary()
print('Task 3, ', model1.summary())

# Plot the scatter plot from Task 1 together with the fitted linear function.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(whr.Freedom, whr["Happiness Score"])
ax.plot(whr.Freedom, task1.olsYhat, color='#F34235')
ax.set(xlabel='Freedom', ylabel='Happiness Score',
       title='Task 3, Happiness Score predicted by Freedom')
plt.show()

# Compute the in-sample (training) mean square error (l2 loss,
# see Assignment 1) with respect to the true values of the response Y.
model1_l2_loss = LP_regression.SquareLoss(task1.xtrain, task1.ytrain, (model1.params).reshape(-1,1))
print('Task 3, losses using sm OLS:\n', model1_l2_loss)
###########################################################################


###########################################################################
# Task 4 : Train a linear regression model (see Assignment 1) using the
# whole sample and the regressor X. In order to estimate parameters
# beta, use first the batch gradient descent method.
# We have provided you with the following starter
# code that runs gradient descent with a particular
# learning rate (alpha). The variable stop_crit is a vector with two
# components: the first indicates if you want to use the value of the l1 norm
# of the gradient as a stopping criterion, the second indicates if you want to use
# the number of iterations as a stopping criterion. You are free to change
# these criteria or to leave them as they are. If both components are set
# to non-zero then both are used and the algorithm stops when one of them
# triggers.
theta_initial1 = [0, 0]  # set initial value
alpha1 = 0.998  # set learning rate, use the optimial one from following code
tolerate1 = 0.000001
maxiter1 = 15000

task1.BGD(theta_initial1, alpha1, tolerate1, maxiter1)
print('Task 4, coefficients using gradient descent:\n', task1.bgdBetahat)  # [[3.79527656][4.2805871 ]]

# Implemement a Gridsearch to for optimizing the learning rate alpha.
time_elapsed = np.zeros(500)
learningrate = np.linspace(0.001, 1, 500)
for n, m in enumerate(learningrate):
    temp = LP_regression(input_x, input_y, 1)
    time_start = time.time()
    temp.BGD(theta_initial1, m, 0.00001, 1500000000)  # set maxiterate very big
    time_end = time.time()
    time_elapsed[n] = time_end - time_start

print('Task 4, optimal learning rate that gives coefficients close to normal equation in the shortest time:',
      learningrate[list(time_elapsed).index(min(time_elapsed))])

# Plot the scatter plot from Task 1 together with the fitted linear function.
y_pred_BGD_train, y_pred_BGD_test = task1.predict('BGD')

fig, ax = plt.subplots()
ax.scatter(whr.Freedom, whr["Happiness Score"], label = 'True score', c = '0.75')
ax.scatter(whr.Freedom, y_pred_BGD_train, label = 'Predicted score', c = '0.1')
ax.plot(whr.Freedom, y_pred_BGD_train, label = 'Regression line', c = 'r')
ax.set_xlabel('Freedom')
ax.set_ylabel('Happiness Score')
ax.set_title('Task 4, Happiness Score predicted by Freedom (using gradient descent)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
plt.show()

# Compute the in-sample (training) mean square error (l2 loss,
# see Assignment 1) with respect to the true values of the response Y.
###########################################################################
print('Task 4, loss (using gradient descent):\n', task1.performance(method='OLS'))


###########################################################################
# Task 5 : Train a linear regression model (see Assignment 1) using the
# whole sample and the regressor X. In order to estimate parameters
# beta, use first the online (stochastic) gradient descent method.
# Your task is to first modify gradientDescentStochastic.
task1.SGD(theta_initial1, 200)
print('Task 5, coefficients using stochastic gradient descent:\n', task1.sgdBetahat)  # array([[4.04331295], [3.6267929 ]])

# Plot the scatter plot from Task 1 together with the fitted linear function.
y_pred_SGD_train, y_pred_SGD_test = task1.predict('SGD')

fig, ax = plt.subplots()
ax.scatter(whr.Freedom, whr["Happiness Score"], label = 'True score', c = '0.75')
ax.scatter(whr.Freedom, y_pred_SGD_train, label = 'Predicted score', c = '0.1')
ax.plot(whr.Freedom, y_pred_SGD_train, label = 'Regression line', c = 'r')
ax.set_xlabel('Freedom')
ax.set_ylabel('Happiness Score')
ax.set_title('Task 5, Happiness Score predicted by Freedom (using stochastic gradient descent)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
plt.show()

# Compute the in-sample (training) mean square error (l2 loss,
# see Assignment 1) with respect to the true values of the response Y.
print('Task 5, loss (using stochastic gradient descent):\n', task1.performance(method='SGD'))
#          Fullsample Loss
# Absolute	0.397819
# Square	0.447556
# Huber	    0.407899

# K-fold cross validation function
crossCeff, crossPerformance = task1.KFoldXV(5, method='SGD')
print('Task 5, coefficients (using stochastic gradient descent and 5-fold cross validation):\n',
      sum(crossCeff.values())/5)
print('Task 5, average loss (using stochastic gradient descent and 5-fold cross validation):\n',
      sum(crossPerformance.values())/5)

# Compare the results
# The loss is almost the same.
###########################################################################



## Part III: Polynomial Regression with One Regressor =====================

###########################################################################
# Task 6 : Construct k = 10 polynomial regression models (nonlinear models).
k = 10
input_y = np.asmatrix(whr["Happiness Score"]).reshape(-1, )
# constrcut a matrix to store all X (x^0 to x^10)
poly_Xmatrix = np.ones(whr.shape[0]).reshape(-1, 1)
for i in range(k):
    poly_Xmatrix = np.hstack([poly_Xmatrix,
                              np.asarray(whr.Freedom).reshape(-1, 1)**(i+1)])

# To train each of the k models, use normal equations to solve for the
# corresponding vector of parameters beta. For each of the k models compute
# the values of the happiness score and the associated mean square error (l2
# loss, see Assignment 1) with respect to the true values of the response Y.
# Illustrate the results plotting k figures each containing the scatter plot
# from Task 1 and the corresponding fitted regression function.
###########################################################################
ols_poly_coeffs = {}
ols_poly_sqloss = {}

fig, axes = plt.subplots(figsize=(22, 18), sharex=True)
axes.axis('off')
fig.suptitle('Task 6, Happiness Score predicted by Freedom and its polynomials (using normal equation)')
for i in range(k):
    input_xi = poly_Xmatrix[:, 0:(i+2)]
    modelname = 'k = '+str(i+1)
    olsreg = LP_regression(input_xi, input_y, 1)
    olsreg.OLS()  # estimate coefficients with OLS
    ols_poly_coeffs[modelname] = olsreg.olsBetahat
    ols_poly_sqloss[modelname] = olsreg.performance(method='OLS').values[1]
    ax = fig.add_subplot(3, 4, i+1)
    x = olsreg.xtrain[:, 1]
    y_pr = olsreg.olsYhat
    x_ordered = x[x.argsort()]
    y_ordered = y_pr[x.argsort()]
    ax.scatter(x, olsreg.ytrain)
    ax.scatter(x, y_pr, color='#F67770')
    ax.plot(x_ordered, y_ordered, linestyle='-', c = 'r')
    ax.set(xlabel='Freedom', ylabel='Happiness Score', title = modelname)
    ax = {}
plt.show()

print('Task 6, coefficients:\n', ols_poly_coeffs)
print('Task 6, l2 losses:\n', pd.DataFrame(ols_poly_sqloss))


###########################################################################
# Task 7 : To train each of the k models, use first the online gradient descent method
# and the functions that you constructed in Task 5.

# Try running gradient descent with different values of alpha and of num_iter and
# see which one gives you the "best" result. (Gridsearch)

# For each of the k models compute
# the values of the happiness score and the associated mean square error (l2
# loss, see Assignment 1) with respect to the true values of the response Y.
# Illustrate the results plotting k figures (you can create subfigures)
# each containing the scatter plot
# from Task 1 and the corresponding fitted regression function.
# Plot the mean square errors for each of the k models together with the ones
# obtained with the normal equations and with the batch gradient descent.
###########################################################################
sgd_poly_coeffs = {}
sgd_poly_sqloss = {}

fig, axes = plt.subplots(figsize=(22, 18), sharex=True)
axes.axis('off')
fig.suptitle('Task 7, Happiness Score predicted by Freedom and its polynomials (using stochastic gradient descent)')
for i in range(k):
    input_xi = poly_Xmatrix[:, 0:(i+2)]
    modelname = 'k = '+str(i+1)
    sgdreg = LP_regression(input_xi, input_y, 1)  # initialize class
    theta = np.random.randn(input_xi.shape[1], 1)
    sgdreg.SGD(theta, 200)
    sgd_poly_coeffs[modelname] = sgdreg.sgdBetahat
    sgd_poly_sqloss[modelname] = sgdreg.performance(method='SGD').values[1]
    ax = fig.add_subplot(3, 4, i+1)
    x = sgdreg.xtrain[:, 1]
    y_pr = sgdreg.sgdYhat
    x_ordered = x[x.argsort()]
    y_ordered = y_pr[x.argsort()]
    ax.scatter(x, sgdreg.ytrain)
    ax.scatter(x, y_pr, color='#F67770')
    ax.plot(x_ordered, y_ordered, linestyle='-', c = 'r')
    ax.set(xlabel='Freedom', ylabel='Happiness Score', title = modelname)
    ax = {}
plt.show()

print('Task 7, coefficients:\n', sgd_poly_coeffs)
print('Task 7, l2 losses:\n', pd.DataFrame(sgd_poly_sqloss))


## Part IV: Regression with two regressors ===============================

###########################################################################
# Task 8 : Use the "Freedom" score and the "Family" score
# as regressors and again let the explained variable (output, response) Y be
# "HappinessScore". First, normalize the features.
input_y2 = np.asmatrix(whr["Happiness Score"]).reshape(-1, 1)
X_2reg = np.hstack([np.ones(whr.shape[0]).reshape(-1, 1),
                      np.asarray(whr.Family).reshape(-1, 1),
                      np.asarray(whr.Freedom).reshape(-1, 1)])

def featureNormalize(X):
    num_features = X.shape[1]
    X_norm = np.zeros(np.shape(X))
    X_norm[:,0] = 1
    for f in range(1,num_features):
        feature = X[:,f]
        mu = feature.mean()
        sigma = feature.std()
        feature_norm = (feature - mu)/sigma
        X_norm[:,f] = feature_norm
    return X_norm

X_norm = featureNormalize(X_2reg)

# Plot the scatter 3d plot. Use all available countries.
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_norm[:, 1], X_norm[:, 2], input_y2, color='r')
ax.set_xlabel('Family')
ax.set_ylabel('Freedom')
ax.set_zlabel('Happiness Score')
ax.set_title('Task 8, Happiness Score (features Freedom and Family are normalized)')
plt.show()

# Train a linear regression model in two variables (regressors) using the normal equations
two_regression = LP_regression(X_norm, input_y2, 1)
two_regression.OLS()
print('Task 8, coefficients using normal equation:\n', two_regression.olsBetahat)

# and then using the batch gradient descent (take what you did in Task 4).
theta_initial2 = [0, 0, 0]  # set initial value
alpha2 = 0.01  # set learning rate, don't set it > 0.01
tolerate2 = 0.000001
maxiter2 = 10000

two_regression.BGD(theta_initial2, alpha2, tolerate2, maxiter2)
print('Task 8, coefficients using gradient descent:\n', two_regression.bgdBetahat)

# Plot the two scatter 3d plots together with the fitted surface (from normal equations).
# normal equation
family_surf, freedom_surf = np.meshgrid(
    np.linspace(X_norm[:, 1].min(), X_norm[:, 1].max(), 200),
    np.linspace(X_norm[:, 2].min(), X_norm[:, 2].max(), 200))
X_surf = np.asmatrix([family_surf.ravel(), freedom_surf.ravel()])
Y_fitsurf = (two_regression.olsBetahat[0] + (two_regression.olsBetahat[1:3]).T @ X_surf)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_norm[:, 1], X_norm[:, 2], input_y2, 'o', color='r')
ax.plot_surface(family_surf, freedom_surf, Y_fitsurf.reshape(family_surf.shape),
                color='#4BAE4F', alpha=0.36)
ax.set_xlabel("Family")
ax.set_ylabel("Freedom")
ax.set_zlabel("Hapiness")
ax.title.set_text(('Task 8, Happiness Score predicted by Freedom and Family (using normal equation)'))
plt.show()

# BGD
Y_fitsurf = (two_regression.bgdBetahat[0] + (two_regression.bgdBetahat[1:3]).T @ X_surf)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_norm[:, 1], X_norm[:, 2], input_y2, 'o', color='r')
ax.plot_surface(family_surf, freedom_surf, Y_fitsurf.reshape(family_surf.shape),
                color='#4BAE4F', alpha=0.36)
ax.set_xlabel("Family")
ax.set_ylabel("Freedom")
ax.set_zlabel("Hapiness")
ax.title.set_text(('Task 8, Happiness Score predicted by Freedom and Family (using Gradient Descent)'))
plt.show()

# Compute the in-sample (training) mean square error (l2 loss,
# see Assignment 1) with respect to the true values of the response Y.
print('Task 8, loss (using normal equation):\n', two_regression.performance(method='OLS'))
print('Task 8, loss (using gradient descent):\n', two_regression.performance(method='BGD'))
###########################################################################




###########################################################################
# Task 9 : Take a regression function as a degree-2 polynomial in two
# variables: freedom and family.
X_2reg_poly = np.hstack([np.ones(whr.shape[0]).reshape(-1, 1),
                      np.asarray(whr.Family).reshape(-1, 1),
                      np.asarray(whr.Freedom).reshape(-1, 1)])

X_2reg_poly = np.append(X_2reg_poly, np.array((whr.Family)**2).reshape(-1,1), axis = 1)
X_2reg_poly = np.append(X_2reg_poly, np.array((whr.Freedom)**2).reshape(-1,1), axis = 1)

# In order to estimate parameters beta, use first the normal equations.
two_reg_poly = LP_regression(X_2reg_poly, input_y2, 1)
two_reg_poly.OLS()
print('Task 9, coefficients using normal equation:\n', two_reg_poly.olsBetahat)

# Compute the in-sample (training) mean square error (l2 loss,
# see Assignment 1) with respect to the true values of the response Y.
print('Task 9, loss (using normal equation):\n', two_reg_poly.performance(method='OLS'))

###########################################################################
# End of Code

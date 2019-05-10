# Supervised Learning: Linear Methods for Regression
# ELS - Chapter 3
# @ Michael

# All codes follows the Chapter 3 from ELS

# Least square estimation
# Of course, we can write a function like the following:
# def ols(Y, X):
#     '''
#     Input:
#     X: a m by n dimension matrix without row and columns names
#     Y: a Y by 1 dimension vector without row and columns names
#     Output:
#     beta: least square estimated coefficients
#     sigma: standard deviation
#     tratio: t-test results
#     ftest: F-test results
#     '''
# However, you will realize that we have to return so many values
# How do we solve this problem?
# We employ the class in python, which you can take it as a bunch of
# functions and attributes

import os
import numpy as np
import pandas as pd
import sys

# get the current working directory and change it to the target one
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/ELS')


# 3.1 Linear Regression Models and Least Squares
# write a class to do least square estimation
class OLS(object):
    '''
    A class for doing least square estimation;
    Rigit now, it only works for the continuous depedent variable;
    the categorical and ordered categorical dependent variable function
    will be added later.
    '''

    def __init__(self, Y, X):
        '''
        Initilize the input: (order matters)
        Y - n by 1 vector
        X - n by m matrix
        Taken X and Y as dataframe format, but still trying to convernt to
        array and matrix
        Default model includes interecpet
        '''
        self.X = X
        self.Y = Y
        self.columns = self.X.columns
        try:
            self.X = np.asarray(self.X)
            self.Y = np.asarray(self.Y)
        except Exception:
            print('There is an error with the input data.')
            sys.exit(0)

        # get the degree of freedom
        self.N = np.shape(self.X)[0]
        self.K = np.shape(self.X)[1]
        YN = np.shape(self.Y)[0]
        if (self.N != YN):
            print('Input Y and X are nonconfortable')
            sys.exit(0)

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

    def estimate(self, normalized=False, standardized=False):
        '''
        Estimate the coefficients, standard error, Z score
        with normalize X or standardize X
        '''
        # normalize or standardize
        if normalized is True and standardized is True:
            print("You can either only normalize X or standardize X,\
                  but not at the same time: dont't set them true at\
                  the same time")
            sys.exit(0)
        elif normalized is True:
            self.X = np.apply_along_axis(OLS.normalize,
                                         axis=0, arr=self.X)
        elif standardized is True:
            self.X = np.apply_along_axis(OLS.standard,
                                         axis=0, arr=self.X)

        # add the intercept vector
        self.intercept = np.ones([self.N, 1])
        self.X = np.hstack([self.intercept, self.X])  # horizontal stack

        # use betahat = inv(X'X)X'Y to get the results
        self.dof = self.N - self.K
        self.betahat = np.linalg.inv(
            np.transpose(self.X) @ self.X) @ np.transpose(self.X) @ self.Y
        # estimation for beta
        # Calculate the standard errors
        self.yhat = self.X @ self.betahat
        self.residu = self.Y - self.yhat
        self.sigmasqr = np.sum(self.residu * self.residu) / self.dof
        self.vmat = self.sigmasqr * np.linalg.inv(np.transpose(self.X) @self.X)
        self.se = np.sqrt(self.vmat.diagonal())
        self.zscore = self.betahat / self.se
        # calculate the F test
        self.ybar = np.mean(self.Y)
        self.xbar = np.mean(self.X, axis=0)
        self.ESS = np.sum((self.yhat - self.ybar)**2)
        self.TSS = np.sum((self.Y - self.ybar)**2)
        self.fscore = (self.ESS / self.K) / (self.TSS / self.dof)

    # define a function to print the tidy results
    def table(self):
        self.betahat = self.betahat.reshape(-1, 1)
        self.se = self.se.reshape(-1, 1)
        self.zscore = self.zscore.reshape(-1, 1)
        results = np.hstack([self.betahat, self.se, self.zscore])
        results = np.around(results, 4)  # use 4 digits
        indexnames = ['Intercept']
        indexnames.extend(self.columns.values)
        self.table = pd.DataFrame(results,
                                  index=indexnames,
                                  columns=['Coefficient',
                                           'Std. Error', 'Z Score'])
        print(self.table)


# read prostate dataset and run regression with default package
prostate = pd.read_csv('prostate.csv', delim_whitespace=True)
# prostate.info()
# prostate.head()
# prostate.columns
# # Index(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45',
# # 'lpsa', 'train']

# preprare the data
pro_train = prostate[prostate.train == 'T']
# pro_train.describe()
pro_x = pro_train.loc[:, 'lcavol':'pgg45'].astype(float)  # convert to float
# pro_x.shape  # check the dimension
pro_y = pro_train.loc[:, 'lpsa']

# Fit the mode
pro_ols = OLS(pro_y, pro_x)  # initialize the class
pro_ols.estimate(standardized=True)
pro_ols.table()

# 3.3 Subset Selection
# LS has two problems: 1) low bias but large variance; 2)interpretation
# best-subset selection is to find the subset of size k that gives smallest
# residual sum of squares
# trade-off: bias and variance; benchmark-AIC(or BIC)

# 3.4 shrinkage methods
# imposing penalty on their size






















# End of Code

# A OLS class for estimating linear model

import numpy as np
import pandas as pd
import sys


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
            print('Input Y and X are noncomfortable')
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

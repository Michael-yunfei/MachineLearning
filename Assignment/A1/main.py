# Assignment 1a, programming with using classes
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
from mpl_toolkits import mplot3d
import statsmodels.api as sm

# get and set the working directory
# please set your own working directory which inlcudes the dataset
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
###############################################################################
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
                            - alpha * (1/n)
                            * self.xtrain.transpose() @ (fx - self.ytrain))
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

    def predict(self):
        insample_ypredic = self.xtrain @ self.coeff
        outsample_ypredic = self.xtest @ self.coeff

        return(insample_ypredic, outsample_ypredic)

    def trainplot2D(self, xlab):
        '''
        Predicted weight values (regression line), predicted data points
        and ground-truth data points for the training data.
        '''
        insample_ypredic = self.xtrain @ self.coeff
        outsample_ypredic = self.xtest @ self.coeff

        xdomain1 = np.linspace(0, np.max(self.xtrain))
        xfit = np.hstack([np.ones(xdomain1.shape[0]).reshape(-1, 1),
                             np.asarray(xdomain1).reshape(-1, 1)])
        yfitline1 = xfit @ self.coeff
        xdomain2 = np.linspace(0, np.max(self.xtest))
        xfit2 = np.hstack([np.ones(xdomain2.shape[0]).reshape(-1, 1),
                             np.asarray(xdomain2).reshape(-1, 1)])
        yfitline2 = xfit2 @ self.coeff
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            'Ground-truth data, predicted data, and fitted regression line')
        axes[0].plot(self.xtrain[:, 1], self.ytrain, 'o', color='#4688F1',
                     label='Groud-truth')
        axes[0].plot(self.xtrain[:, 1], insample_ypredic, 'o', color='#F34235',
                     label='Predicted')
        axes[0].plot(xdomain1, yfitline1, color='#F67770',
                     label='Fitted line', linewidth=1.5)
        axes[0].set(xlabel=xlab, ylabel='Weight',
                    title='Insample')
        axes[0].legend(loc=4)
        axes[1].plot(self.xtest[:, 1], self.ytest, 'o', color='#4688F1',
                     label='Groud-truth')
        axes[1].plot(self.xtest[:, 1], outsample_ypredic, 'o', color='#F34235',
                     label='Predicted')
        axes[1].plot(xdomain2, yfitline2, color='#F6776F',
                     label='Fitted line', linewidth=1.5)
        axes[1].set(xlabel=xlab, ylabel='Weight',
                    title='Outsample')
        axes[1].legend(loc=4)
        plt.show()
###############################################################################


###############################################################################
# Single variable regression
###############################################################################

# initialize the class
# make insure input are in matrix format and add constant
input_x = np.hstack([np.ones(MandMs.shape[0]).reshape(-1, 1),
                     np.asarray(MandMs.Red).reshape(-1, 1)])

input_y = np.asmatrix(MandMs['Weight']).reshape(-1, 1)
theta_initial = [0, 0]  # set initial value
alpha = 0.01  # set learning rate, don't set it > 0.01
tolerate = 0.000001
maxiter = 15000

gradientEs1 = ML_gradient(input_x, input_y, theta_initial, 0.8,
                          randomsplit=True)
gradientEs1.estimate(alpha, tolerate, maxiter)
gradientEs1.coeff  # evenytime will be differnt as sample is slected randomly

# check results with python package
# model = sm.OLS(gradientEs1.ytrain, gradientEs1.xtrain).fit()
# model.summary()

gradientEs1.performance()
gradientEs1.trainplot2D('Red')


###############################################################################
# Two variables regression
###############################################################################

# initialize the class
input_x2 = np.hstack([np.ones(
    MandMs.shape[0]).reshape(-1, 1),
                      np.asarray([MandMs.Green, MandMs.Blue]).reshape(-1, 2)])
theta_initial2 = [0, 0, 0]
gradientEs2 = ML_gradient(input_x2, input_y, theta_initial2, 0.8,
                          randomsplit=True)
gradientEs2.estimate(alpha, tolerate, maxiter)
gradientEs2.coeff  # evenytime will be differnt as sample is slected randomly

# check results with python package
# model = sm.OLS(gradientEs2.ytrain, gradientEs2.xtrain).fit()
# model.summary()  #it's same value in 4 digits

gradientEs2.performance()

# you can get the predict values if you want
insample_predict, outsample_predict = gradientEs2.predict()


# 3D plot
green_surf, blue_surf = np.meshgrid(
    np.linspace(MandMs.Green.min(), MandMs.Green.max(), 100),
    np.linspace(MandMs.Blue.min(), MandMs.Blue.max(), 100))
X_surf = np.asmatrix([green_surf.ravel(), blue_surf.ravel()]).reshape(-1, 2)
Y_fitsurf = gradientEs2.coeff[0] + X_surf @ gradientEs2.coeff[1:3]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gradientEs2.xtrain[:, 1], gradientEs2.xtrain[:, 2],
           gradientEs2.ytrain, 'o')
ax.plot_surface(green_surf, blue_surf, Y_fitsurf.reshape(green_surf.shape),
                color='None', alpha=0.3)
ax.set_xlabel("Green")
ax.set_ylabel("Blue")
ax.set_zlabel("Weight")
plt.show()


###############################################################################
# compare performace with last 20% (outsample)
###############################################################################

# nonrandomly selected

gradientEs1_nonrandom = ML_gradient(input_x, input_y, theta_initial, 0.8)
gradientEs1_nonrandom.estimate(alpha, tolerate, maxiter)
gradientEs1_nonrandom.performance()
#       Insample Loss	Outsample Loss
# Absolute	0.528365	0.720710
# Square	0.953769	1.406060
# Huber	    0.684693	1.006636
gradientEs1_nonrandom.trainplot2D('red')


gradientEs2_nonrandom = ML_gradient(input_x2, input_y, theta_initial2, 0.8)
gradientEs2_nonrandom.estimate(alpha, tolerate, maxiter)
gradientEs2_nonrandom.performance()

#       Insample Loss	Outsample Loss
# Absolute	0.554169	0.885200
# Square	0.922815	1.958244
# Huber	    0.700217	1.295105

# Model 1 is better in terms of outsample fit
# End of Code

# Gradient descent: regression example
# @ Michael

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# check the working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/GraidentDescent')

# read the dataset
ex1data = pd.read_csv('Ex1data.csv', names=['Population', 'Profit'])

# explore the dataset
ex1data.head()
ex1data.columns
ex1data.shape  # (97, 2)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(ex1data.Population, ex1data.Profit, marker='x')
ax.set(xlabel='Population of City in 10,000s',
       ylabel='Profit in $10,000s',
       title='Plot of X and Y')
fig.show()


# run regression: use population to predict profit

# set x
datamatrix = np.asmatrix(ex1data['Population']).transpose()
datamatrix.shape
input_x = np.hstack([np.ones(ex1data.shape[0]).reshape(-1, 1), datamatrix])
input_x.shape  # check the matrix shape, (97, 2)
input_y = np.asmatrix(ex1data.Profit).transpose()
input_y.shape


# define the loss function
def SquareLoss(x, y, theta):
    """
    A square error loss function to compute the loss (cost) error
    Input: x - matrix n by m
           y - vector n by 1
           theta - vector m by 1, order matters considering the intercept
    Output: the sum of squared error
    """
    n = x.shape[0]
    fx = x @ theta  # matrix (dot) production
    loss = 1/2 * 1/n * np.sum(np.square(fx - y))  # use average with 1/n

    return(loss)


# set intial theta value
theta_initial = np.array([0, 0]).reshape(-1, 1)
# test SquareLoss function
SquareLoss(input_x, input_y, theta_initial)  # 32.072733877455676


# define the gradient descent function
def GradientDescent(x, y, theta, alpha, tolerate, maxiterate):
    i = 0  # set the iteration counting index
    tolerate_rule = 1  # set the initial tolerate rate
    n = x.shape[0]
    current_theta = theta
    cost_vector = np.empty([0, 1])

    # iterate
    while tolerate_rule >= tolerate and i <= maxiterate:
        sl = np.array(SquareLoss(x, y, current_theta)).reshape([1, 1])
        cost_vector = np.append(cost_vector, sl, axis=0)  # store cost function
        fx = x @ current_theta
        update_theta = current_theta - alpha * (1/n) * x.transpose() @ (fx - y)
        tolerate_rule = np.minimum(np.abs(update_theta - current_theta))
        i += 1
        current_theta = update_theta

    return(current_theta, cost_vector)


theta_initial = np.array([0, 0]).reshape(-1, 1)  # give initial value
alpha = 0.01  # learning rate
tolerate = 0.00001  # tolerate rates
maxiter1 = 1500
coefficents1, lossvalues1 = GradientDescent(input_x, input_y,
                                            theta_initial, alpha,
                                            tolerate, maxiter1)

print("The estimated coefficients are", coefficents1)
# The estimated coefficients are [[-3.63077001] [ 1.16641043]]
lossvalues1.shape
# iteration stops because function reaches to maxiter, (1501, 1)
plt.plot(lossvalues1[1:])

# we can set maxiter = 3000 to see what's going on
theta_initial = np.array([0, 0]).reshape(-1, 1)  # give initial value
alpha = 0.01  # learning rate
tolerate = 0.00001  # tolerate rates
maxiter2 = 3000
coefficents2, lossvalues2 = GradientDescent(input_x, input_y,
                                            theta_initial, alpha,
                                            tolerate, maxiter2)

print("The estimated coefficients are", coefficents2)
# The estimated coefficients are [[-3.84072806][ 1.18750299]]


lossvalues2.shape  # (2372, 1), iteration stops because of tolerate

# plot the cost function
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(lossvalues2[1:])
ax.set(title='Plot of Loss Function', xlabel='Iteration',
       ylabel='Loss')
fig.show()


# plot the regression line
xdomain = np.linspace(5, 25)
yfit = coefficents2[0] + coefficents2[1] * xdomain

fig, ax = plt.subplots(figsize=(6, 5), sharex=True)
ax.scatter(ex1data.Population, ex1data.Profit, marker='x',
           label='Raw data')
ax.plot(xdomain.reshape(-1, 1), yfit.reshape(-1, 1), 'r',
        label='Linear regression (Gradient descent)')
ax.set(xlabel='Population of City in 10,000s',
       ylabel='Profit in $10,000s',
       title='Plot of X and Y with fitted regression')
ax.legend(loc=4)
fig.show()


# show the dynamics of loss function in gradient descent
# I will add it later
# End of code

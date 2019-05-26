%  The basic structure of the exercise is taken from Andrew Ng

%  Linear and polynomial regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you to get started on the
%  polynomial regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     gradientDescent.m
%     computeLossLinModel.m
%     computeLoss.m
%     gradientDescentStochastic.m
%     featureNormalize.m
%     normalEqn.m
%

%%%%%%          Due to : 29.05.2019 23:59           %%%%%%%%%%%%%%%%%%%%%%%

%% Part I: Load Data ======================================================
%  Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

% Load All Data discarding the first three columns containing Country, 
% Region, HappinessRank 

data_file = csvread('WHR2016.csv', 1, 3);
fprintf('Data successfully loaded.\n')

%% Part II: Regression with one regressor =================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 1 : Create a regressor (input, feature) X using the "Freedom" score and  
% an explained variable (output, response) Y from "HappinessScore". 
% Plot the scatter plot of both variables. Use all available countries.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 2 : Train a linear regression model (see Assignment 1) using the
% whole sample and the created regressor X. In order to estimate parameters
% beta, use first the normal equations. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 3 : Check the computations with 
% built-in regress function (for example the function regress in matlab).
% Plot the scatter plot from Task 1 together with the fitted linear 
% function. Compute the in-sample (training) mean square error (l2 loss,
% see Assignment 1) with respect to the true values of the response Y.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 4 : Train a linear regression model (see Assignment 1) using the
% whole sample and the regressor X. In order to estimate parameters
% beta, use first the batch gradient descent method. 
% We have provided you with the following starter
% code that runs gradient descent with a particular
% learning rate (alpha). The variable stop_crit is a vector with two
% components: the first indicates if you want to use the value of the l1 norm 
% of the gradient as a stopping criterion, the second indicates if you want to use
% the number of iterations as a stopping criterion. You are free to change
% these criteria or to leave them as they are. If both components are set
% to non-zero then both are used and the algorithm stops when one of them
% triggers.
%
% Implemement a Gridsearch to for optimizing the learning rate alpha.
%
% Plot the scatter plot from Task 1 together with the fitted linear 
% function. Compute the in-sample (training) mean square error (l2 loss,
% see Assignment 1) with respect to the true values of the response Y.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 5 : Train a linear regression model (see Assignment 1) using the
% whole sample and the regressor X. In order to estimate parameters
% beta, use first the online (stochastic) gradient descent method. 
% Your task is to first modify gradientDescentStochastic.
%
% Implement a gridsearch for "optimizing" the parameters num_iter and alpha.
%
% Plot the scatter plot from Task 1 together with the fitted linear 
% function. Compute the in-sample (training) mean square error (l2 loss,
% see Assignment 1) with respect to the true values of the response Y.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% Part III: Polynomial Regression with One Regressor =====================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 6 : Construct k = 10 polynomial regression models (nonlinear models). 
% Regression function is a polynomial function of one variable X ("Freedom" 
% score), consider monomial basis (see Tutorial 2 example). 
% To train each of the k models, use normal equations to solve for the
% corresponding vector of parameters beta. For each of the k models compute
% the values of the happiness score and the associated mean square error (l2
% loss, see Assignment 1) with respect to the true values of the response Y.
% Illustrate the results plotting k figures each containing the scatter plot
% from Task 1 and the corresponding fitted regression function. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 7 : To train each of the k models, use first the online gradient descent method
% and the functions that you constructed in Task 5.
%
% Try running gradient descent with different values of alpha and of num_iter and
% see which one gives you the "best" result. (Gridsearch)
%
% For each of the k models compute
% the values of the happiness score and the associated mean square error (l2
% loss, see Assignment 1) with respect to the true values of the response Y.
% Illustrate the results plotting k figures (you can create subfigures) 
% each containing the scatter plot
% from Task 1 and the corresponding fitted regression function. 
% Plot the mean square errors for each of the k models together with the ones 
% obtained with the normal equations and with the batch gradient descent.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%% Part IV: Regression with two regressors ===============================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 8 : Use the "Freedom" score and the "Family" score
% as regressors and again let the explained variable (output, response) Y be 
% "HappinessScore". First, normalize the features. For that complete 
% the function featureNormalize. Plot the scatter 3d plot. Use all available countries.
% Train a linear regression model in two variables (regressors) using the 
% normal equations and then using the batch gradient descent (take what you did in Task 4). 
% Compare the parameter values (in l2 norm).
% Plot the two scatter 3d plots together with the fitted surface (from normal equations). 
% Compute the in-sample (training) mean square error (l2 loss,
% see Assignment 1) with respect to the true values of the response Y.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 9 : Take a regression function as a degree-2 polynomial in two
% variables: freedom and family. In order to estimate parameters
% beta, use first the normal equations. 
% Plot the scatter plot from Task 8 together with the fitted surface. 
% Compute the in-sample (training) mean square error (l2 loss,
% see Assignment 1) with respect to the true values of the response Y.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






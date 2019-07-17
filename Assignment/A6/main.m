% Machine Learning Assignment 6 at University of Konstanz 2019
% due to 26.07.19

% The goal of this exercise is finding the weights in a neural network with the
% architecture used in the previous exercise that minimize the empirical classification error
% in the handwritting recognition problem.
% This will be done by implementing the backpropagation algorithm.

clc; clearvars; close all

% Read the data out of the mentioned files and create a design matrix X
% in which the rows contain the pixel gray levels of each image. Each row
% should contain 400 values. Create also a vector y containing the labels
% associated to each picture

X = csvread('digits_data.csv');
y = csvread('digits_labels.csv');

% Your Task: 

% Create a function [J grad] = nn_twolayer_loglikely(nn_params, input_layer_size, ...
% hidden_layer_size, num_labels, X, y, lambda) that computes the log-likelihood
% of a neural network with one hidden layer as well as its gradient using
% the backpropagation algorithm

% Create a function x=sigmoid(z) which calculates the sigmoid of z

% Check the correctness of you implementation by comparing the gradient output
% of your function nnlogLikelihood to its numerical evaluation using the 
% definition of the derivative. Perform this check using the weights 
% Theta1 and Theta2 stored in the file 'WeightsNN.mat'. It is enough to
% compute the first 100 elements

load('WeightsNN.mat');

nn_params = [Theta1(:) ; Theta2(:)];
input_layer_size  = 400; 
hidden_layer_size = 25;  
num_labels = 10;
n = length(y);
lambda = 0;

[J grad] = nn_twolayer_loglikely(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda);

% We compute now the numerical gradient

numgrad = zeros(size(nn_params));
perturb = zeros(size(nn_params));
epsilon = 1e-4;
for p = 1:100
    % Set perturbation vector
    perturb(p) = epsilon;
    J_perturb = nn_twolayer_loglikely(nn_params + perturb, input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda);
    % Compute Numerical Gradient
    numgrad(p) = (J_perturb - J) / epsilon;
    perturb(p) = 0;
end
Error_gradient = norm(numgrad(1:100)-grad(1:100))/norm(numgrad(1:100)+grad(1:100));
if Error_gradient>10e-2
    warning('your gradient is not correct')
end

% Use the function nn_twolayler_loglikely to train the neural network to the classification
% of the digits in 'digits_data.csv'. Use the weights 
% Theta1 and Theta2 stored in the file 'WeightsNN.mat' as initial values in
% the optimization process.

% Feel free to optimize the layersize and the regulasation parameter lambda
% if you want to. 
lambda = 1;
costFunction = @(params) nn_twolayer_loglikely(params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                          
options = optimoptions(@fminunc,'GradObj','on','Algorithm','quasi-newton', 'Display', 'iter', 'MaxIter', 50);
[nn_params_backprop, cost] = fminunc(costFunction, nn_params, options);
% For those who code in python or R, use an inbuild optizimizer or code it
% by yourself. What ever you feel like.

% Use the NN obtained to classify the digits in 'digits_data.csv' and calculate the error
% the classifcation vector should be called y_classified_nn

% Your Code


% end

% Create a misclassification matrix whose (i,j)th component
% denotes the percentage of times in which the classifier sends the 
% figure with label i to the class j.
num_labels = 10;
misclassification_matrix_nn = zeros(num_labels, num_labels);

% Your code



% end

% Use the function displayData in order to visualize some missclassified data
% Not necessary if you code in python or R

% Examples:
% sevens that get classified as a nine.
X_7_to_9 = X(find((y == 7) & (y_classified_nn == 9)), :);
displayData(X_7_to_9);
% threes that get classified as an eight.
X_3_to_8 = X(find((y == 3) & (y_classified_nn == 8)), :);
displayData(X_3_to_8);
% threes that get classified as an eight.
X_1_to_4 = X(find((y == 1) & (y_classified_nn == 4)), :);
displayData(X_1_to_4);
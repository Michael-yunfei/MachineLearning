function [mu_params, cov_params, pi_params] = fitLDA(featuresSet, groupSet, classLabels)
%
%  Input
%  -----
%
%  featuresSet: where each row corresponds to an observation or replicate, 
%  and each column corresponds to a feature or variable 
%
%  groupSet: variable with each row representing a class label. 
%  Each element of groupSet specifies the group of the corresponding row of 
%  featuresSet
%
% classLabels: provide the labels according to which we order the output
% parameters. They determine in which order the parameters are constructed
%
%  Output
%  mu_params: contains the learnt (estimated) mean values for the given
%  feature within the classes/groups. In the case of binary classification and 1 feature
%  case mu_params should be a vector with two real components. Otherwise -
%  you can organize them for example as m x k matrix where each j-th column
%  (with j = {1,...,k})
%  contains means for all m features
%
%  cov_params: contains the learnt (estimated) covariance shared by all the  
%  the classes/groups (in the 1 feature
%  case cov_params should contain just the variance value). Otherwise - 
%  covariance matrix in dimension m 
%
%  pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
%  (see slide 25 in L5_ML) for each i-th class (with i = {1,...,k}). 
%  In the case of binary classification
%  case pi_params should be a vector with two nonnegative values with sum = 1
%
%  Instructions
%  ------------
%  Note that these are supervised learning algorithms, thus:
%   - pi_params are sample-based estimates of prior probabilities of each class;
%   - mu_params are means of features within teh classes;
%   - cov_params is the pooled mean of the var/covariance matrix shared by
%     all classes (see slide 31 in L5_ML);
%

% ====================== YOUR CODE STARTS HERE ======================

% ====================== YOUR CODE ENDS HERE ========================

end


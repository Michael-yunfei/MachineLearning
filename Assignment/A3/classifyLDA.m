function [groupSet, xCritValue] = classifyLDA(featuresSet, mu_params, cov_params, pi_params, classLabels, thresholdValue, computeCritValue)
%
%  Input
%  -----
%
%  featuresSet: variable where each row corresponds to an observation or replicate
%  to be classified, and each column corresponds to a feature or input variable. 
%
%  mu_params: contains the learnt (estimated) mean values for the given
%  feature within the classes/groups. In the case of binary classification and 1 feature
%  case mu_params should be a vector with two real components. 
%  Otherwise:
%  you can organize them for example as m x k matrix where each j-th column
%  (with j = {1,...,k}) contains means for all m features.
%
%  cov_params: contains the learnt (estimated) covariance shared by all the  
%  the classes/groups (in the 1 feauture case cov_params should contain 
%  just the variance value). 
%  Otherwise: covariance matrix in dimension m.
%
%  pi_params: contains the learnt (estimated) P(Y=label) prior probabilities
%  (see slide 25 in L5_ML) for each i-th class (with i = {1,...,k}). 
%  In the case of binary classification case pi_params should be a vector 
%  with two nonnegative values with sum = 1.
%
%  classLabels: labels assigned to featuresSet; groupSet will be labeled
%  with classLabels.
%
%  thresholdValue: can be specified for the case of binary classification. The 
%  default value is 1 which corresponds to the case of
%  \delta_1(x)=\delta_0(x)
%  This case is considered in slide 31 in L5_ML for
%  one feature instance (hint: be attentive with the log transform,
%  log(1) = 0 for the thresshold).
%
%  computeCritValue: valid only for binary classification and just 1
%  feature, is set to 0 by default; 
%
%  Output:
%  groupSet: contains labels for each of the input featuresSet
%
%  xCritValue: if computeCritValue is different from 0 contains the
%  boundary value (slide 31 in L5_ML) for a given thresholdValue

% =========================================================================
%  The code below should classify featuresSet:
%
%  FOR TASK 1: for the binary classification and one feature you can use explicit expression for 
%  critical values as decision boundaries; however, you will still need to generalize 
%  the expressions in slide 31 in L5_ML for the case of threshold different from 1.
%  FOR OTHER TASKS: multiple classes and/or multiple features you will have
%  to compute the Gaussian discriminant functions delta_r for r = {1,...,k}
%  see slide 30 in L5_ML 


% Determine if it is a binary classification task. If yes, use critical
% value.
xCritValue = NaN;
classLabelsNumber = length(classLabels);
[numObs, numFeatures] = size(featuresSet);

if classLabelsNumber == 2
        % in the following code allow for the different thresholds to be set

    % ====================== YOUR CODE STARTS HERE ======================
    
    % ====================== YOUR CODE ENDS HERE ========================

    if computeCritValue && numFeatures == 1
        
        % ====================== YOUR CODE STARTS HERE ======================

        
        % ====================== YOUR CODE ENDS HERE ========================
        
    end

else
    
    % ====================== YOUR CODE STARTS HERE ======================
    

    
    % ====================== YOUR CODE ENDS HERE ========================
    
end


end


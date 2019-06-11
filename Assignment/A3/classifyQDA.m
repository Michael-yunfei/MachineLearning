function groupSet = classifyQDA(featuresSet, mu_params, cov_params, pi_params, classLabels, thresholdValue)
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
%  cov_params: contains the learnt (estimated) covariances for each of  
%  the k classes/groups (in the 1 feature
%  case cov_params should contain just the k variance values). Otherwise - 
%  covariance matrices in dimension m. You can think of creating a cell
%  array or a three-dimensional array
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
%
%
%  The code below should classify featuresSet according to QDA.
%

classLabelsNumber = length(classLabels);
[numObs, numFeatures] = size(featuresSet);

delta_vals = zeros(numObs, classLabelsNumber);

% compute the delta_vals
    % ====================== YOUR CODE STARTS HERE ======================

    % ====================== YOUR CODE ENDS HERE ========================

% if one has only two classes you can chose a different threshold value
% (other than 1)

if classLabelsNumber == 2 && thresholdValue ~= 1

    % ====================== YOUR CODE STARTS HERE ======================

    % ====================== YOUR CODE ENDS HERE ========================

else
    
    % ====================== YOUR CODE STARTS HERE ======================
    
 
    % ====================== YOUR CODE ENDS HERE ========================
    
end


end


function groupSet = classifyLogistic(featuresSet, logit_param)
%
%  Input
%  -----
%
%  featuresSet: variable where each row corresponds to an observation or replicate
%  to be classified, and each column corresponds to a feature or input variable. 
%
%  logit_param: contains the learnt (estimated) values for the logit 'beta'
%  for the given feature within the classes/groups.
%
%  The code below should classify featuresSet according to Logit.
%

[numObs, ~] = size(featuresSet);
[~, d] = size(logit_param);

% ====================== YOUR CODE STARTS HERE ======================


% ====================== YOUR CODE ENDS HERE ========================

end


function [L, TIError, TIIError] = computeLoss(Y, Y_pred, typeOfLoss)
%COMPUTELOSS Compute loss 

% Initialize some useful values

% You need to return the following variables correctly 
L = 0;

% ------------------------------------
% relevant only for binary classification. In this case the labels are 
% assumed to be 1 and 0.
TIError  = 0;
TIIError = 0;
% ------------------------------------

if strcmp(typeOfLoss, 'mse')
        % ====================== YOUR CODE STARTS HERE ====================
        
      
        % ====================== YOUR CODE ENDS HERE ======================
        
elseif strcmp(typeOfLoss, '0-1')
    
    if isequal(sort(unique(Y_pred)), [0; 1])
        
        % ====================== YOUR CODE STARTS HERE ====================
        
      
        % ====================== YOUR CODE ENDS HERE ======================
        
    else
        
        % ====================== YOUR CODE STARTS HERE ====================

        
        % ====================== YOUR CODE ENDS HERE ======================
        
    end
    
else
    
    fprintf('the type of loss is not known');
    
end


end

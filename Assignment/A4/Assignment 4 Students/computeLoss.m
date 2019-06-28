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
    
    L = mean((Y(:) - Y_pred(:)).^2);

elseif strcmp(typeOfLoss, '0-1')
    
    if isequal(sort(unique(Y_pred)), [0; 1])
        
        % ====================== YOUR CODE STARTS HERE ====================
        
        TIError  = length(find(Y_pred(:) == 1 & Y(:) == 0));
        TIIError = length(find(Y_pred(:) == 0 & Y(:) == 1));
        
        L = (TIError + TIIError)/length(Y);
        
        % ====================== YOUR CODE ENDS HERE ======================
        
    else
        
        % ====================== YOUR CODE STARTS HERE ====================
        
        L = sum(Y_pred(:) ~= Y(:))/length(Y);
        
        % ====================== YOUR CODE ENDS HERE ======================
        
    end
    
elseif strcmp(typeOfLoss, 'hinge')
            
    L = sum(max(0, 1-Y.*Y_pred))/length(Y);
    
else
    
    fprintf('the type of loss is not known');
    
end


end

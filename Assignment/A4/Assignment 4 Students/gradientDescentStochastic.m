function [beta_path, beta, L_history] = gradientDescentStochastic(X, Y, beta, alpha, num_iter)
% GRADIENTDESCENT Performs online gradient descent to learn beta
%   beta = GRADIENTDESCENT(X, Y, beta, alpha, num_iters) updates beta by
%   stochastic gradient steps with learning rate alpha

% Initialize some useful values

[n, d] = size(X);
L_history = 0;
beta_path = beta(:)';
iter = 1;
numClasses = length(unique(Y));

% ====================== YOUR CODE STARTS HERE ======================

rng('shuffle');

if numClasses == 2
    
    while iter < num_iter
        n1 = randperm(n);
        X = X(n1, :);
        Y = Y(n1);
        for i = 1:n
            % compute the gradient:
            
            grad = -(Y(i) - 1/(1 + exp(- X(i, :) * beta))) * X(i, :);
            grad = grad';
            % update:
            beta = beta - alpha * grad;
            beta_path = [beta_path; beta(:)'];
        end
        % Save the cost L in every iteration
        % computeLoss(Y, Y_pred, typeOfLoss)
        % L_history(iter) = computeLossLinModel(X, Y, beta);
        iter = iter + 1;
    end
    
else
    
    beta = reshape(beta, d, numClasses - 1);
    
    while iter < num_iter
        n1 = randperm(n);
        X = X(n1, :);
        Y = Y(n1);
        grad = zeros(d, numClasses - 1);
        for i = 1:n
            temp = 0;
            temp_num = zeros(d, numClasses - 1);
            for j = 1:numClasses - 1
                temp = temp + exp(beta(:, j)' * X(i, :)');
                grad(:, j) = grad(:, j) + (Y(i) == j) * X(i, :)';
                temp_num(:, j) = X(i, :)' * exp(beta(:, j)' * X(i, :)');
            end
            grad = grad - temp_num./(1 + temp); 
        end
        grad = - grad / n; % regularize gradient
        % update:
        beta = beta - alpha * grad; 
        beta_path = [beta_path; beta(:)'];
        iter = iter + 1;
    end
    
end

% ====================== YOUR CODE ENDS HERE ======================

end

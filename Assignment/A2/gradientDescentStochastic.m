function [beta_path, beta, L_history] = gradientDescentStochastic(X, Y, beta, alpha, num_iter)
    %GRADIENTDESCENT Performs online gradient descent to learn beta
    %   beta = GRADIENTDESCENT(X, Y, beta, alpha, num_iters) updates beta by 
    %   stochastic gradient steps with learning rate alpha

    % Initialize some useful values
    n = length(Y); % number of training examples
    L_history = 0;
    beta_path = beta(:)';
    iter = 1;
    rng('shuffle');







end

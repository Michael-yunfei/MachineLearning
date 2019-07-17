function [J grad] = nn_twolayer_loglikely(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%Computes the neural network cost function and its gradient for a two layer
%neural network which performs classification using backpropagation.
% nn_params contains Theta1 and Theta2 in vectorized form. Also the
% gradient is provided in vectorized form and needs to be
% converted back into the weight matrices.


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));
n = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Your code



%end


end

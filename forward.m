% Forward propagation function
function Y_pred = forward(X, W, b, num_hidden_layers, relu)
    % Initialize input layer
    
    % Hidden layer computation
    for i = 1:num_hidden_layers
        Z = X * W{i} + b{i};     % Compute weighted input
        X = relu(Z);             % Apply activation function
    end
    
    % Output layer computation
    Y_pred = X * W{num_hidden_layers + 1} + b{num_hidden_layers + 1};
end
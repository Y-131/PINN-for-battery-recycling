%% Initialization
clear;clc;close;
rng(0.01)
%% Parameter settings
input_size = 6;                                  % Number of input neurons (features)
hidden_size = 32;                                % Number of neurons in each hidden layer
output_size = 1;                                 % Number of output neurons
num_hidden_layers = 2;                           % Number of hidden layers
learning_rate = 0.015;                           % Learning rate for optimization
epochs = 1000;                                   % Number of training epochs

I = 8.4;                                         % Pulse current (in Amperes)
l1 = 0.95;                                       % Loss function weight for main target
l2 = 0.045;                                      % Loss function weight for R0
l3 = 0.005;                                      % Loss function weight for Rc-C expression
%% Loading data
data = xlsread('Train_batch1.xlsx');             % Load pulse response dataset
model = xlsread('batch1_model.xlsx');            % Load equivalent circuit model parameters

%% Physical information model
x = model(:,7);                                  % SOH
y = model(:,1);                                  % OCV
z1 = model(:,8);                                 % R0
z2 = model(:,9);                                 % RC
z3 = model(:,10);                                % C

%% Train/test set split
total_indices = 1:550;                           % Index range for the dataset

train_start = 1;
test_start = 1;
test_end = 55;                                   % Parameters for training and test set split

% Extract training indices
train_indices = [total_indices(train_start:test_start-1), total_indices(test_end+1:end)];
test_indices = total_indices(test_start:test_end);

% Columns for inputs and outputs
input_cols = 1:6;
output_col = 7;

% Construct training data
PN = data(train_indices, input_cols)';
TN = data(train_indices, output_col)';

% Construct test data
PM = data(test_indices, input_cols)';
TM = data(test_indices, output_col)';


%% Data normalization
% Normalize PN to [0,1]
[pn, ps_input] = mapminmax(PN, 0, 1);            
pn = pn';

% Apply same normalization to PM
pm = mapminmax('apply', PM, ps_input);           
pm = pm';

% Normalize TN
[tn, ps_output] = mapminmax(TN, 0, 1);          
tn = tn';

X_train = pn;                                    % Final training inputs
Y_train = tn;                                    % Final training outputs

%% Weight and bias initialization
W = cell(1, num_hidden_layers + 1);
b = cell(1, num_hidden_layers + 1);

% Input to first hidden layer
W{1} = randn(input_size, hidden_size) * sqrt(2 / input_size);
b{1} = randn(1, hidden_size) * 0.15;

% Hidden to hidden layers
for i = 2:num_hidden_layers
    W{i} = randn(hidden_size, hidden_size) * sqrt(2 / hidden_size);
    b{i} = randn(1, hidden_size) * 0.15;
end

% Output layer
W{num_hidden_layers + 1} = randn(hidden_size, output_size) * sqrt(2 / hidden_size);
b{num_hidden_layers + 1} = randn(1, output_size) * 0.15;

%% Activation function: tanh (approximate)
Tanh = @(x) (2 ./ (1 + exp(-2*x)) - 1);         % Activation function
Tanh_prime = @(x) (1 - Tanh(x).^2);             % Derivative of activation

%% Loss function: Custom MSE + physical penalty terms
mse_loss = @(Y_true, Y_pred, U1_true, U1_pred, U2_true, U2_pred) ...
    l1*0.5*mean((Y_true - Y_pred).^2) + ...
    l2*0.5*ones(1,output_size)*mean((U1_true - U1_pred).^2) + ...
    l3*0.5*ones(1,output_size)*mean((U2_true - U2_pred).^2);

%% Adam optimizer parameters
beta1 = 0.986;                                  % Decay rate for 1st moment
beta2 = 0.985;                                  % Decay rate for 2nd moment
epsilon = 1e-8;                                 % Small constant to avoid division by zero

% Initialize moments
m_W = cell(1, num_hidden_layers + 1);
v_W = cell(1, num_hidden_layers + 1);
m_b = cell(1, num_hidden_layers + 1);
v_b = cell(1, num_hidden_layers + 1);

for i = 1:num_hidden_layers + 1
    m_W{i} = zeros(size(W{i}));
    v_W{i} = zeros(size(W{i}));
    m_b{i} = zeros(size(b{i}));
    v_b{i} = zeros(size(b{i}));
end

%% Neural network training loop
for epoch = 1:epochs
    total_loss = 0;                             % Reset loss for this epoch

    % Forward pass
    A = cell(1, num_hidden_layers + 1);         % Activations
    Z = cell(1, num_hidden_layers + 1);         % Weighted inputs

    Z{1} = X_train * W{1} + b{1};
    A{1} = Tanh(Z{1});
    for j = 2:num_hidden_layers
        Z{j} = A{j-1} * W{j} + b{j};
        A{j} = Tanh(Z{j});
    end
    Z{num_hidden_layers + 1} = A{num_hidden_layers} * W{num_hidden_layers + 1} + b{num_hidden_layers + 1};
    A{num_hidden_layers + 1} = Z{num_hidden_layers + 1}; 

    % Reverse normalization to get physical values
    Yj = forward(X_train, W, b, num_hidden_layers, Tanh);
    Phyj = mapminmax('reverse', Yj, ps_output)';

    % Interpolate physical parameters from physical model
    R0 = griddata(x, y, z1, Phyj(:,1), PN(1,:)', 'nearest');
    Rc = griddata(x, y, z2, Phyj(:,1), PN(1,:)', 'nearest');
    C  = griddata(x, y, z3, Phyj(:,1), PN(1,:)', 'nearest');

    % Compute loss including physical consistency penalties
    loss_mse = mse_loss(Y_train, A{num_hidden_layers + 1}, ...
                        PN(4,:)', I*R0, ...
                        PN(3,:)', I*Rc.*(exp(-1./(Rc.*C))-exp(-41./(Rc.*C))));
    total_loss = total_loss + loss_mse;

    % Backpropagation
    dA = (l1*(A{num_hidden_layers + 1} - Y_train) + ...
          ones(495,output_size)*l2.*(PN(4,:)'-I*R0) + ...
          ones(495,output_size)*l3.*(PN(3,:)'-I*Rc.*(exp(-1./(Rc.*C))-exp(-41./(Rc.*C))))) ...
          / size(X_train, 1);
    dZ = dA;

    % Backprop and Adam update
    for j = num_hidden_layers + 1:-1:2
        dW = A{j-1}' * dZ;
        db = sum(dZ, 1);

        m_W{j} = beta1 * m_W{j} + (1 - beta1) * dW;
        v_W{j} = beta2 * v_W{j} + (1 - beta2) * (dW .^ 2);
        m_b{j} = beta1 * m_b{j} + (1 - beta1) * db;
        v_b{j} = beta2 * v_b{j} + (1 - beta2) * (db .^ 2);

        m_W_hat = m_W{j} / (1 - beta1^epoch);
        v_W_hat = v_W{j} / (1 - beta2^epoch);
        m_b_hat = m_b{j} / (1 - beta1^epoch);
        v_b_hat = v_b{j} / (1 - beta2^epoch);

        W{j} = W{j} - learning_rate * m_W_hat ./ (sqrt(v_W_hat) + epsilon);
        b{j} = b{j} - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + epsilon);

        dZ = dZ * W{j}' .* Tanh_prime(Z{j-1});
    end

    % Update first layer
    dW = X_train' * dZ;
    db = sum(dZ, 1);
    W{1} = W{1} - learning_rate * dW;
    b{1} = b{1} - learning_rate * db;

    % Display loss every 100 epochs
    if mod(epoch, 100) == 0
        disp(['Epoch ' num2str(epoch) ': Loss = ' num2str(total_loss)]);
    end
end

%% Prediction using trained model

Y_test = forward(X_train, W, b, num_hidden_layers, Tanh)';       % Prediction on training set
Y = mapminmax('reverse', Y_test, ps_output);                     % Denormalize

Y_1 = forward(pm, W, b, num_hidden_layers, Tanh);                % Prediction on test set
Y_1 = Y_1';
Y1 = mapminmax('reverse', Y_1, ps_output);                       % Denormalize


%% Evaluation metric for test set

error = Y1 - TM;
R2 = 1 - sum((TM - Y1).^2) / sum((mean(TM) - TM).^2);
MSE = mean(error.^2);
RMSE = sqrt(MSE);
MAPE = (100 / length(TM)) * sum(abs(error ./ Y1));
MAE = mean(abs(error));

disp(['MSE：', num2str(MSE)]);
disp(['RMSE：', num2str(RMSE)]);
disp(['MAPE：', num2str(MAPE)]);
disp(['MAE：', num2str(MAE)]);
disp(['R²：', num2str(R2)]);

plot(TM,Y1,'bo')

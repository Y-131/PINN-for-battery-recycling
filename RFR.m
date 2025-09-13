%% Initialization
clc
clear
close all

%% Loading data
data = xlsread('Train_batch1.xlsx'); % Import dataset from Excel file

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
[pn, ps_input] = mapminmax(PN, 0, 1); % Normalize input data to (0,1)
pn = pn';

pm = mapminmax('apply', PM, ps_input); % Apply the same normalization to test set
pm = pm';

[tn, ps_output] = mapminmax(TN, 0, 1); % Normalize output data
tn = tn';

%% Model parameter settings and training
trees = 200; % Number of decision trees set to 200
leaf  = 0.8; % Minimum leaf size set to 0.8
OOBPrediction = 'on';  % Enable out-of-bag error estimation
OOBPredictorImportance = 'on'; % Compute predictor (feature) importance
Method = 'regression';  % Choose between regression or classification (regression selected)
net = TreeBagger(trees, pn, tn, 'OOBPredictorImportance', OOBPredictorImportance, ...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);

%% Model prediction
pyuce = predict(net, pm );

%% Reverse normalization
Pyuce = mapminmax('reverse', pyuce, ps_output); % Restore predicted test output
Pyuce = Pyuce';

%% Performance metrics calculation
error = Pyuce - TM;
[~, len] = size(TM);
R2 = 1 - sum((TM - Pyuce).^2) / sum((mean(TM) - TM).^2); % Coefficient of determination
MSE = error * error' / len;  % Mean Squared Error
MAE = sum(abs(error)) / len; % Mean Absolute Error
RMSE = sqrt(MSE);            % Root Mean Squared Error
MAPE = 100 * sum(abs(error ./ Pyuce)) / len; % Mean Absolute Percentage Error

disp(['MSE of test data: ', num2str(MSE)])
disp(['MAE of test data: ', num2str(MAE)])
disp(['RMSE of test data: ', num2str(RMSE)])
disp(['MAPE of test data: ', num2str(MAPE)])
disp(['R^2 of test data: ', num2str(R2)])

plot(TM,Pyuce,'bo')

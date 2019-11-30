clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10

% Load Training Data
fprintf('Loading Data Set...\n')

load('data1.mat');
m = size(X, 1);

randSelection = randperm(size(X, 1));
randSelection = randSelection(1:100);

displayData(X(randSelection, :));

fprintf('\nLoading Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2

load('weights.mat');

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Computing Cost

% Weight regularization parameter
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Also output the costFunction debugging values

debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

%% Find Most Optimal Lambda
%% lambda = optimumLambda(input_layer_size, ...
%%                       hidden_layer_size, ...
%%                       num_labels, X, y, initial_nn_params)

%% Using Most Optimal Lambda (Found At 0.1)

fprintf('\nTraining Neural Network... \n')
lambda = 0.1;
options = optimset('MaxIter', 400);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Reshape Theta1 and Theta2 back from nn_params

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


fprintf('Displaying Hidden Layer \n');

displayData(Theta1(:, 2:end));

fprintf('Displaying Output Layer \n')

displayData(Theta2(:, 2:end));

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('\nPress Enter To Check Out Some Examples\n');
pause

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display Example

    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

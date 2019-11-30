function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

  % Reshape nn_params back into the parameters Theta1 and Theta2

  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  m = size(X, 1);

  % You need to return the following variables correctly
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  % Map Y To 5000x10 Matrix Of Vectors
  y_matrix = eye(num_labels)(y,:);

  % Implementing Feedforward Propagation

  num_layers = 2;

  a1 = [ones(m,1) X];
  z2 = a1*Theta1';
  a2 = [ones(m,1) sigmoid(z2)];
  z3 = a2*Theta2';
  h = sigmoid(z3);


  J = 1/m * ( ( -y_matrix.*log(h) ) - ( (1-y_matrix).*log(1 - h) ) );
  J = sum(J(:));

  % Regularisation

  t1Sq = Theta1(:, 2:end).^2;
  t2Sq = Theta2(:, 2:end).^2;
  reg = (lambda/(2*m)) * ( sum( t1Sq(:) ) + sum( t2Sq(:) ) );

  J = J + reg;

  % Back Propagation

  d3 = h - y_matrix;
  Delta2 = d3' * a2;

  d2 = ( d3 * Theta2(:, 2:end) ) .* sigmoidGradient(z2);
  Delta1 = d2' * a1;

  Theta1(:,1) = 0;
  Theta2(:,1) = 0;

  Theta1_grad = Delta1/m + (lambda/m)*Theta1;
  Theta2_grad = Delta2/m + (lambda/m)*Theta2;

  % Unroll gradients

  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

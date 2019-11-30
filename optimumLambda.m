function lambda = optimumLambda(input_layer_size, ...
                               hidden_layer_size, ...
                               num_labels, X, y, initial_nn_params)

  %% Find most optimal lambda

  options = optimset('MaxIter', 20);
  lambda = [0.1, 0.3, 1, 3, 10];
  costs = zeros(length(lambda), 1);

  for count = 1:length(lambda)
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda(count));

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    costs(count) = cost(length(cost));

  end

  [val, ind] = min(costs);
  fprintf("lowest cost: %f ",val);
  fprintf("at lambda: %f \n",lambda(ind));

end

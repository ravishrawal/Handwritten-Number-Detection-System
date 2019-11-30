function W = randInitializeWeights(L_in, L_out)

  eps_in = sqrt(6)/sqrt(L_in + L_out);
  W = rand(L_out, 1 + L_in) * (2 * eps_in) - eps_in;

end

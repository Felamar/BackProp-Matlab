clc
clear
close
cases = [      % XOR
        1 1 0; % Case 1
        1 0 1; % Case 2
        0 1 1; % Case 3
        0 0 0  % Case 4
    ];
rng(69) % seed for random number generator for reproducibility
nn = neural_network();  
for i = 1:4
    fprintf('\nTraining for the given case: %d\n', i);
    nn = nn.BackPropagation(cases(i, :));
    
    prediction = nn.Predict(cases(i, 1:2));
    fprintf('\nPrediction for the given case: %d\n', prediction);
end
% nn = nn.BackPropagation(cases(i, :));



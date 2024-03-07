clc
cases = [      % XOR
        1 1 0; % Case 1
        1 0 1; % Case 2
        0 1 1; % Case 3
        0 0 0  % Case 4
    ];
rng(69) % seed for random number generator for reproducibility
i = input("Case no: ");
fprintf('Training the neural network with the case: [%d %d]\nExpected value: %d\n\n', cases(i, :));
nn = neural_network();  
nn = nn.BackPropagation(cases(i, :));
prediction = nn.Predict(cases(i, 1:2));

fprintf('\nPrediction for the given case: %d\n', prediction);


cases = [ % XOR
        1    1    1e-5; 
        1    1e-5 1; 
        1e-5 1    1;
        1e-5 1e-5 1e-5
    ];

i = input("case no:");
nn = neural_network();  
nn = nn.BackPropagation(cases(i, :));

prediction = nn.Predict(cases(i, 1:2));
disp(prediction);

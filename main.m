cases = [ % XOR
        1    1    1e-5; 
        1    1e-5 1; 
        1e-5 1    1;
        1e-5 1e-5 1e-5
    ];
rng(69); 

nn = neural_network();  
nn = nn.BackPropagation(cases);

prediction = nn.Predict(cases(1,1:2));
disp(prediction);

cases = [ % XOR
        1 1 0; 
        1 0 1; 
        0 1 1;
        0 0 0
    ];
rng(69); 

nn = neural_network();
nn = nn.BackPropagation(cases);
nn = nn.ForwardPropagation(cases(1,1:2));
% nn = nn.ForwardPropagation(cases(2,1:2));
% nn = nn.ForwardPropagation(cases(3,1:2));
% nn = nn.ForwardPropagation(cases(4,1:2));

disp(nn.activations(1, 3));

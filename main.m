cases = [ % XOR
        1 1 0; 
        1 0 1; 
        0 1 1;
        0 0 0
    ];
rng(69); 

nn = neural_network();
% nn = nn.ForwardPropagation([1 1]);
nn = nn.BackPropagation(cases);
nn = nn.ForwardPropagation(cases(4,1:2));
disp(nn.output);

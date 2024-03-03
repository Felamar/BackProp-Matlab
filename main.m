inputs  = [1 1];
neurons = 3;
bias    = [1 1 1];
outputs = 1;
target  = 0;
rng(69); 

nn = neural_network(inputs, neurons, bias, outputs);
nn.weight_matrix_array
% nn = nn.forward_propagation();
nn = ForwardPropagation(nn);
neurons = nn.neurons;
BackPropagation(nn, target);

for j = 1:size(neurons, 2)
    for i = 1:size(neurons, 1)
        fprintf('%f    ',neurons(i,j).activation);
    end
    fprintf('\n');
end
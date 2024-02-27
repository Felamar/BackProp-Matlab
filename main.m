inputs  = [1 1];
neurons = 3;
outputs = 1;
target  = 0;
rng(123); 

nn = neural_network(inputs, neurons, outputs);
nn = nn.forward_propagation();
neurons = nn.neurons;
for j = 1:size(neurons, 1)
    for i = 1:size(neurons, 2)
        fprintf('%f    ',neurons(i,j).Activation);
    end
    fprintf('\n');
end
function nn = ForwardPropagation(neural_network)
    neurons_aux = neural_network.neurons;
    for n_matrix = 1:size(neural_network.sizes, 2)-1
        for i = 1:neural_network.sizes(n_matrix+1)
            z_l = 0;
            for j = 1:neural_network.sizes(n_matrix)
                z_l = z_l + neurons_aux(n_matrix, j).activation * neural_network.weight_matrix_array(j, i, n_matrix);
            end
            z_l = z_l + neural_network.bias(n_matrix);
            neurons_aux(n_matrix+1, i).z_l = z_l;
            neurons_aux(n_matrix+1, i).activation = sigmoid_f(z_l);	
        end
    end
    neural_network.neurons = neurons_aux;
    nn = neural_network;
end
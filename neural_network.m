classdef neural_network
    properties (Access = public)
        sizes
        neurons
        bias
        weight_matrix_array
    end

    methods
        function obj = neural_network(inputs, hid_size, bias, out_size)
            obj.bias  = bias;
            obj.sizes = [size(inputs, 2) size(inputs, 2), hid_size, out_size];
            neurons(4, 3) = neuron();
            for i=1:4
                for j=1:obj.sizes(i)
                    if i == 1
                        neurons(i, j) = neuron(inputs(j));
                    else
                        neurons(i, j) = neuron();
                    end
                end
            end
            obj.neurons = neurons;

            for k = 1:size(obj.sizes, 2)-1
                for i = 1:obj.sizes(k)
                    for j = 1:obj.sizes(k+1)
                        obj.weight_matrix_array(i, j, k) = rand();
                    end
                end
            end
        end
    end
end
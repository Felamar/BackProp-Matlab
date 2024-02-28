classdef neural_network
    properties (Access = public)

        input_layer_size
        hidden_layer_size
        output_layer_size

        neurons
        bias
        input_to_hidden_w
        hid_to_hid2_w
        hidden_to_output_w
        target
        weight_matrix_array
    end

    methods
        function obj = neural_network(inputs, hid_size, bias, out_size, target)
            obj.input_layer_size = size(inputs, 2);
            obj.hidden_layer_size = hid_size;
            obj.output_layer_size = out_size;
            obj.bias   = bias;
            obj.target = target;
            sizes = [obj.input_layer_size, obj.input_layer_size, obj.hidden_layer_size, obj.output_layer_size];
            neurons(4, 3) = neuron();
            for i=1:4
                for j=1:sizes(i)
                    if i == 1
                        neurons(i, j) = neuron(inputs(j));
                    else
                        neurons(i, j) = neuron();
                    end
                end
            end
            obj.neurons = neurons;

            
            input_to_hidden_w = zeros(size(inputs, 2), obj.input_layer_size);
            for i = 1:size(inputs, 2)
                for j = 1:obj.input_layer_size
                    input_to_hidden_w(i,j) = rand();
                end
            end
            obj.input_to_hidden_w = input_to_hidden_w;

            hid_to_hid2_w = zeros(size(inputs, 2), hid_size);
            for i = 1:obj.input_layer_size
                for j = 1:hid_size
                    hid_to_hid2_w(i,j) = rand();
                end
            end
            obj.hid_to_hid2_w = hid_to_hid2_w;

            hidden_to_output_w = zeros(hid_size, out_size);
            for i = 1:hid_size
                for j = 1:out_size
                    hidden_to_output_w(i,j) = rand();
                end
            end
            obj.hidden_to_output_w = hidden_to_output_w;

        end

        function obj = forward_propagation(obj)
            % sigmoid_f = @(x) 1 / (1 + exp(-x));

            neurons_aux = obj.neurons;

            for j=1:obj.input_layer_size
                z_l=0;
                for k=1:obj.input_layer_size
                    z_l = z_l + neurons_aux(1,k).Activation * obj.input_to_hidden_w(k,j);
                end
                z_l=z_l+obj.bias(1);
                neurons_aux(2,j).z_l        = z_l;
                neurons_aux(2,j).Activation = sigmoid_f(z_l);
            end

            for j=1:obj.hidden_layer_size
                z_l = 0;
                for k=1:obj.input_layer_size
                    z_l = z_l + neurons_aux(2,k).Activation * obj.hid_to_hid2_w(k,j);
                end
                z_l=z_l+obj.bias(2);
                neurons_aux(3,j).z_l        = z_l;
                neurons_aux(3,j).Activation = sigmoid_f(z_l);
            end

            for j=1:obj.output_layer_size
                z_l = 0;
                for k=1:obj.hidden_layer_size
                    z_l = z_l + neurons_aux(3,k).Activation * obj.hidden_to_output_w(k,j);
                end
                z_l=z_l+obj.bias(3);
                neurons_aux(4,j).z_l        = z_l;
                neurons_aux(4,j).Activation = sigmoid_f(z_l);
            end

            obj.neurons = neurons_aux;
        end

        % function obj = back_propagation(obj, target)
        %     output = obj.neurons(3, 1).Activation
        %     sigmoid_f = @(x) 1 / (1 + exp(-x));
        %     sigmoid_prime_f = @(x) sigmoid_f(x) * (1 - sigmoid_f(x));


            

        % end
    end
end
classdef neural_network
    properties (Access = public)
        input_layer_size
        hidden_layer_size
        output_layer_size

        neurons
        input_to_hidden_w
        hidden_to_output_w
    end

    methods
        function obj = neural_network(inputs, hid_size, out_size)
            obj.input_layer_size = size(inputs, 2);
            obj.hidden_layer_size = hid_size;
            obj.output_layer_size = out_size;
            neurons(3, 3) = neuron();
            for i = 1:size(inputs, 2)
                neurons(1,i) = neuron(inputs(i));
            end
            for i = 1:hid_size
                neurons(2,i) = neuron();
            end
            for i = 1:out_size
                neurons(3,i) = neuron();
            end
            obj.neurons = neurons;

            input_to_hidden_w = zeros(size(inputs, 2), hid_size);
            for i = 1:obj.input_layer_size
                for j = 1:hid_size
                    input_to_hidden_w(i,j) = rand();
                end
            end
            obj.input_to_hidden_w = input_to_hidden_w;

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

            for j=1:obj.hidden_layer_size
                z_l = 0;
                for k=1:obj.input_layer_size
                    z_l = z_l + neurons_aux(1,k).Activation * obj.input_to_hidden_w(k,j);
                end
                neurons_aux(2,j).z_l        = z_l;
                neurons_aux(2,j).Activation = sigmoid_f(z_l);
            end

            for j=1:obj.output_layer_size
                z_l = 0;
                for k=1:obj.hidden_layer_size
                    z_l = z_l + neurons_aux(2,k).Activation * obj.hidden_to_output_w(k,j);
                end
                neurons_aux(3,j).z_l        = z_l;
                neurons_aux(3,j).Activation = sigmoid_f(z_l);
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
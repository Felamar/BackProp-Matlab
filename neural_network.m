classdef neural_network
    properties (Access = public)
        % number of layers in the network (including the input layer)
        n_layers
        % 1D array of layer sizes
        % where the index represents the layer
        szs 
        % 3D array of weight matrices for all layers
        % where the row index represents the neuron in the previous layer
        % and the column index represents the neuron in the current layer
        weights 
        % 1D array of activation vectors for all layers
        % where the index represents the bias of the layer
        bias
        % 2D array of activation vectors for all layers
        % where the row index represents the neuron in the layer
        % and the column index represents the layer
        activations
        % 2D array of z value vectors for all layers
        % where the row index represents the neuron in the layer
        % and the column index represents the layer
        zs
        % a value between 0 and 1
        output
        
    end

    methods
        % inputs = vector de  2 valores
        % hid_size = cantidad de neuronas en la capa oculta
        % bias = vector de 3 valores
        % out_size = cantidad de neuronas en la capa de salida
        function obj = neural_network()
            n_inputs     = 2;      % número de entradas (2 bits)
            hid_sz       = [2, 3]; % número de neuronas en las capas ocultas
            out_sz       = 1;      % número de neuronas en la capa de salida
            obj.n_layers = 4;      % número de capas (incluyendo la de entrada)
            obj.szs      = [n_inputs, hid_sz(1), hid_sz(2), out_sz]; % vector de tamaños de capas

            obj.weights     = [];                               % matriz de pesos
            obj.bias        = [1 1 1];              % vector de bias
            obj.activations = zeros(max(obj.szs), obj.n_layers-1);  % matriz de activaciones
            obj.zs          = zeros(max(obj.szs), obj.n_layers-1);  % matriz de activaciones

            % Inicialización de la matriz de pesos
            for matrix_i = 2:obj.n_layers
                currentl_sz  = obj.szs(matrix_i);
                previousl_sz = obj.szs(matrix_i-1);
                for row = 1:previousl_sz
                    for col = 1:currentl_sz
                        obj.weights(row, col, matrix_i-1) = rand();
                    end
                end
            end

        end

        function obj = ForwardPropagation(obj, inputs)
            % inputs is a 1D array of input values (2 bits)
            inputs = [inputs 0];
            n_matrix = obj.n_layers - 1;
            for matrix_i = 1:n_matrix
                currentl_sz = obj.szs(matrix_i+1);
                p_layer = matrix_i;
                for neuron_i = 1:currentl_sz
                    if matrix_i == 1
                        z = inputs * obj.weights(:, neuron_i, matrix_i) + obj.bias(matrix_i);
                    else
                        z = obj.activations(:, p_layer)' * obj.weights(:, neuron_i, matrix_i) + obj.bias(matrix_i);
                    end
                    obj.zs(neuron_i,matrix_i) = z;
                    obj.activations(neuron_i, matrix_i) = Sigmoid(z);
                end
            end
        end

        function obj = BackPropagation(obj, cases)
            % input is a 1D array of input values (2 bits)
            % expected is a value of the expected output (XOR gate)
            % input = [1, 1];
            % expected = 0;
            n_iterations = 1e5;
            learning_rate = 0.01;
            o = 3;
            h2 = 2;
            h1 = 1;
            error_2  = 10;
            
            while n_iterations > 0
                av_out_w_costs  = zeros(3, 3);
                av_h2_w_costs   = zeros(3, 3);
                av_h1_w_costs   = zeros(3, 3);
                % Forward propagation
                
                % for case_i = 1:size(cases, 1)
                %     inputs   = cases(case_i, 1:2);
                %     expected = cases(case_i, 3);
                %     obj      = obj.ForwardPropagation(inputs);
                %     error    = obj.activations(1, o) - expected;
                %     error_2  = error^2;
                    


                %     % Backward propagation
                %     % h2 activations x sigmoid prime of z_out x 2error
                %     out_w_costs = obj.activations( : , h2) .* SigmoidPrime(obj.zs(1,o))  * (2*error);
                %     delta_o     = obj.weights(:, 1, o)     .* SigmoidPrime(obj.zs(1,o))  * (2*error);
                    
                %     h2_w_costs  = obj.activations(1:2, h1) * SigmoidPrime(obj.zs(:,2))' .* delta_o';
                %     delta_h2    = obj.weights(1:2, : , h2) * (SigmoidPrime(obj.zs(:,h2)) .* delta_o);

                %     h1_w_costs  = inputs' * SigmoidPrime(obj.zs(1:2,h1))' .* delta_h2';

                %     av_out_w_costs(: ,   1) = av_out_w_costs( : ,   1) + out_w_costs;
                %     av_h2_w_costs(1:2,  : ) = av_h2_w_costs( 1:2,  : ) + h2_w_costs;
                %     av_h1_w_costs(1:2, 1:2) = av_h1_w_costs(1:2, 1:2)  + h1_w_costs;
                %     % pause

                % end

                inputs   = cases(2, 1:2);
                expected = cases(2, 3);
                obj      = obj.ForwardPropagation(inputs);

                error    = obj.activations(1, o) - expected;
                error_2  = error^2;
                


                % Backward propagation
                % h2 activations x sigmoid prime of z_out x 2error
                out_w_costs = obj.activations( : , h2) .* SigmoidPrime(obj.zs(1,o))  * (2*error);
                delta_o     = obj.weights(:, 1, o)     .* SigmoidPrime(obj.zs(1,o))  * (2*error);
                
                h2_w_costs  = obj.activations(1:2, h1) * SigmoidPrime(obj.zs(:,2))' .* delta_o';
                delta_h2    = obj.weights(1:2, : , h2) * (SigmoidPrime(obj.zs(:,h2)) .* delta_o);

                h1_w_costs  = inputs' * SigmoidPrime(obj.zs(1:2,h1))' .* delta_h2';

                av_out_w_costs(: ,   1) = av_out_w_costs( : ,   1) + out_w_costs;
                av_h2_w_costs(1:2,  : ) = av_h2_w_costs( 1:2,  : ) + h2_w_costs;
                av_h1_w_costs(1:2, 1:2) = av_h1_w_costs(1:2, 1:2)  + h1_w_costs;
                % pause


                % av_out_w_costs = av_out_w_costs / 1;
                % av_h2_w_costs  = av_h2_w_costs  / 1;
                % av_h1_w_costs  = av_h1_w_costs  / 1;
                
                obj.weights( : ,   1, o)  = obj.weights( : ,   1, o)  - learning_rate * av_out_w_costs( : ,   1);
                obj.weights(1:2,  : , h2) = obj.weights(1:2,  : , h2) - learning_rate * av_h2_w_costs(1:2,  : );
                obj.weights(1:2, 1:2, h1) = obj.weights(1:2, 1:2, h1) - learning_rate * av_h1_w_costs(1:2, 1:2);
                n_iterations = n_iterations - 1;
            end
            
        end
    end
end
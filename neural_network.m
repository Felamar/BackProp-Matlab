classdef neural_network
    properties (Access = public)
        % Neural Network properties
        n_layers % Number of layers in the network
        szs      % Sizes of the layers
        bias     % Bias vector
        
        % Weights for each layer
        w_h1 
        w_h2
        w_o
        % Activations of the layers
        a_h1
        a_h2
        a_o
        % z values of the layers
        z_h1
        z_h2
        z_o
    end

    methods
        function obj = neural_network()
            n_inputs     = 2;      % Input layer size
            hid_sz       = [2, 3]; % Hidden layers sizes
            out_sz       = 1;      % Output layer size
            obj.n_layers = 4;      % Layers in the network
            obj.szs      = [n_inputs, hid_sz(1), hid_sz(2), out_sz]; % Layer sizes
            obj.bias     = rand(1, 3); % bias vector

            % Weight values for each layer
            obj.w_h1 = rand(2, 2) - 0.5;
            obj.w_h2 = rand(2, 3) - 0.5;
            obj.w_o  = rand(3, 1) - 0.5;

            % z values of the layers
            obj.z_h1 = zeros(2);
            obj.z_h2 = zeros(3);
            obj.z_o  = zeros(1); 

            % Activations of the layers
            obj.a_h1 = zeros(2); 
            obj.a_h2 = zeros(3); 
            obj.a_o  = zeros(1); 
            
        end

        function obj = ForwardPropagation(obj, inputs)
            % Activation * Weights + Bias
            obj.z_h1 = inputs   * obj.w_h1 + obj.bias(1); % 1x2 * 2x2 
            obj.a_h1 = Sigmoid(obj.z_h1);

            obj.z_h2 = obj.a_h1 * obj.w_h2 + obj.bias(2); % 1x2 * 2x3
            obj.a_h2 = Sigmoid(obj.z_h2);

            obj.z_o  = obj.a_h2 * obj.w_o  + obj.bias(3); % 1x3 * 3x1
            obj.a_o  = Sigmoid(obj.z_o);
        end

        function obj = BackPropagation(obj, cases)
            n_iterations = 1e6;
            learning_rate = 1;
            error_2  = zeros(1, n_iterations);
            counter  = 1;

            while counter <= n_iterations
                out_w_costs = zeros(1, 3);
                h2_w_costs  = zeros(2, 3);
                h1_w_costs  = zeros(2, 2);

                for i = 1:4
                    inputs   = cases(i, 1:2);
                    expected = cases(i, 3);

                    % Forward Propagation
                    % -----------------------------------------------------------------------------------------
                    obj   = obj.ForwardPropagation(inputs);
                    error = obj.a_o - expected;
                    error_2(1, counter) = error_2(1, counter) + error^2 / 2;

                    % Gradient Descent
                    % -----------------------------------------------------------------------------------------
    
                    % Activation * SigmoidPrime * Error
                    out_w_costs = out_w_costs + (obj.a_h2 .* SigmoidPrime(obj.z_o) .* (error));
                    delta_o     = obj.w_o  .* SigmoidPrime(obj.z_o) * (error);
    
                    % Activation * SigmoidPrime * Sum(phi_o * w_o)
                    h2_w_costs  = h2_w_costs + (obj.a_h1' * SigmoidPrime(obj.z_h2) .* delta_o');
                    delta_h2    = obj.w_h2  * (SigmoidPrime(obj.z_h2) .* delta_o')';
    
                    % Activation * SigmoidPrime * Sum(phi_h2 * w_h2)
                    h1_w_costs  = h1_w_costs + (inputs' * SigmoidPrime(obj.z_h1) .* delta_h2');
                    
                end
                
                % Update weights
                % -----------------------------------------------------------------------------------------
                obj.w_o(:, :)  = obj.w_o( :, :) - learning_rate .* out_w_costs';
                obj.w_h2(:, :) = obj.w_h2(:, :) - learning_rate .* h2_w_costs;
                obj.w_h1(:, :) = obj.w_h1(:, :) - learning_rate .* h1_w_costs';
                error_2(1, counter) = error_2(1, counter) / 4;
                counter = counter + 1;
            end
            % Plot error
            % -----------------------------------------------------------------------------------------
            figure;
            plot(error_2, 'r');
            title('Error across epochs');
            txt = ['Case: [', num2str(cases(1)), ' ', num2str(cases(2)), '] Expected: ', num2str(floor(cases(3)))];
            subtitle(txt);
            legend('Error');
            xlabel('Epochs');
            ylabel('Error');
            grid on
            set(gcf, 'Position', [100, 100, 1500, 750]);
            xlim([-1e3, n_iterations+1e3]);
            ylim([-0.01, 0.25]);
            disp(['Final error: ', num2str(error_2(end))])
        end

        function output = Predict(obj, inputs)
            obj = obj.ForwardPropagation(inputs);
            output = obj.a_o;
        end
    end
    % floor division

end
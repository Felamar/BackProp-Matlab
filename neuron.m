classdef neuron
    properties (Access = public)
        activation
        z_l
        is_bias
    end

    methods
        function obj = neuron(varargin)
            if nargin == 0
                obj.activation = 0;
            end
            if nargin == 1
                obj.activation = varargin{1};
            end

            if nargin == 2
                obj.activation = varargin{1};
                obj.is_bias    = varargin{2};
            end
        end
    end
end
classdef neuron
    properties (Access = public)
        Activation
        z_l
        is_bias
    end

    methods
        function obj = neuron(varargin)
            if nargin == 0
                obj.Activation = 0;
            end
            if nargin == 1
                obj.Activation = varargin{1};
            end

            if nargin == 2
                obj.Activation = varargin{1};
                obj.is_bias    = varargin{2};
            end
        end
    end
end
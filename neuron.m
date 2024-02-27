classdef neuron
    properties (Access = public)
        Activation
        z_l
    end

    methods
        function obj = neuron(varargin)
            if nargin == 0
                obj.Activation = 0;
            else
                obj.Activation = varargin{1};
            end
        end
    end
end
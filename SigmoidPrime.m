function sp = SigmoidPrime(x)
    sp = Sigmoid(x) .* (1 - Sigmoid(x));  
    % if x > 0
    %     sp = 1;
    % else
    %     sp = 0;
    % end
end
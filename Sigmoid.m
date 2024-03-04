function s = Sigmoid(input)
    s = 1./(1+exp(-input));
    % if input > 0
    %     s = 1;
    % else
    %     s = 0;
    % end
end
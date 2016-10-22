function [y] = logisticFct(X)
% Sigmoid function

    y = zeros(length(X),1);
    neg = find(X<0);
    pos = find(X>=0);

    % In order to take care of numerical errors, we process negative and
    % positive values of X differently.
    y(pos) = exp(-log(1+exp(-X(pos))));
    y(neg) = exp(X(neg)-log(1+exp(X(neg))));
end


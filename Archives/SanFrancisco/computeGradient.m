function g = computeGradient(y,tX,beta)
% Compute the gradient

    e = y - tX*beta;
    N = length(y);
    g = -(1/N)*tX'*e;
end
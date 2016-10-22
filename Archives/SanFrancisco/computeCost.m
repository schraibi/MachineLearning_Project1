function [L] = computeCost(y, tX, beta)
% Compute the MSE given y and yHat = tX*beta
    N = length(y);
    e = y-tX*beta;

    L = e'*e/(2*N);
    
end


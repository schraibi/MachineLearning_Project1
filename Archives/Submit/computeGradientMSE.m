function g = computeGradientMSE(y,tX,beta)
%COMPUTEGRADIENTMSE Computes the gradient of the MSE cost
    % beta should be a row vector
    % tX is length(y)*length(beta) matrix
%   
    N = length(y);
    e = y - tX*beta;
    g = -1/N*tX'*e;
end


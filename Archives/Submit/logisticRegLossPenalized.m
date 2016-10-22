function [L, g, H] = logisticRegLossPenalized(beta, y, tX, lambda)
%LOGISTICREGLOSSPENALIZED Returns the penalized Cost, Gradient and Hessian
%   It adds a lambda parameter which will be in charge of lifting the
%   eigenvalues to avoid problems.
    L = computeCostPenalized(y,tX,beta, lambda);
    g = computeGradientLogistic(y,tX,beta);
    H = computeHessian(y,tX,beta);
end


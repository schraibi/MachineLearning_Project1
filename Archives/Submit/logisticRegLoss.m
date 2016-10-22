function [L, g, H] = logisticRegLoss(beta, y, tX)
%LOGISTICREGLOSS Returns the Cost, Gradient and Hessian
%   Returns the Cost, Gradient and Hessian associated to logistic
    L = computeCostLogistic(y,tX,beta);
    g = computeGradientLogistic(y,tX,beta);
    H = computeHessian(y,tX,beta);
end


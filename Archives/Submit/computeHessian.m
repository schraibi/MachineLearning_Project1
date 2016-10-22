function [ H ] = computeHessian( y,tX,beta )
%COMPUTEHESSIAN Computes the Hessian of the logistic cost
%   Detailed explanation goes here
    pred = tX * beta;
    sigma = exp(pred)./(1+exp(pred));
    S = diag(sigma)*diag(1-sigma);
    H = tX'*S*tX;

end


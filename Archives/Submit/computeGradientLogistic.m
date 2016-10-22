function [ g ] = computeGradientLogistic( y,tX,beta )
%COMPUTEGRADIENTLOGISTIC Computes the gradient of the logistic cost
%   Detailed explanation goes here
pred = tX*beta;
sigma = exp(pred)./(1 + exp(pred));
g = tX'*( sigma - y);

end


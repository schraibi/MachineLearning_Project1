function [ L ] = computeCostMSE( y, tX, beta )
%computeCostMSE Computes the cost associated to Logistic
%   
    e = y -tX*beta;
    L = e' * e/(2*length(y));

end


function [ cost ] = computeCostLogistic( y, tX, beta )
%COMPUTECOSTLOGISTIC Computes the cost associated to Logistic
%   
    pred = tX * beta;
    cost = - (y' * pred - sum(log(1 + exp(pred)))) ;  

end


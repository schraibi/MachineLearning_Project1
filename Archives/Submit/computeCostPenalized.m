function [ penalizedCost ] = computeCostPenalized( y, tX , beta, lambda )
%COMPUTECOSTPENALIZED Computes the cost associated to Logistic
%   
    penalizedCost = computeCostLogistic(y, tX, beta) + lambda * sum(beta(1:length(beta)).*beta(1:length(beta)));
end


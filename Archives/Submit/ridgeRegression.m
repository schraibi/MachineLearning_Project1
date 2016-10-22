function [ beta ] = ridgeRegression( y,tX,lambda )
%RIDGEREGRESSION Implements ridge regression and returns the betas
%   
M = size(tX,2)-1;
LAMBDA = lambda*[zeros(1,M+1);zeros(M,1),eye(M)];
beta = (tX'*tX + LAMBDA)\(tX'*y);
end


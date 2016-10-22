function [ beta ] = ridgeRegression( y,tX, lambda )
%RIDGEREGRESSION is a penalized least squares method

M = size(tX,2)-1;
lam = [0,zeros(1,M);zeros(M,1),lambda*eye(M)]; % lam = [0,    0     ]
beta = ((tX'*tX) + lam)\(tX'*y);               %       [0, lambda*I ]
end


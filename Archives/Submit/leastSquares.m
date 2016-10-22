function [ beta ] = leastSquares( y, tX )
%LEASTSQUARES Implements leastSquares through normal equations
%   We used SVN decomposition to avoid teh ill-conditionning
    %beta = (tX'*tX)\(tX'*y);
    [U,S,V] = svd(tX);
    beta=V*pinv(S)*(U')*y;
end


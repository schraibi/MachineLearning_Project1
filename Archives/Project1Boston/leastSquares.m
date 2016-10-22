function [ beta ] = leastSquares( y, tX )
%LEASTSQUARES Apply the least squares method

beta = (tX'*tX)\(tX'*y);

end
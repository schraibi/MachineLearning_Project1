function [ beta ] = leastSquaresGD( y, tX, alpha )
%LEASTSQUARESGD Gradient descent applied to least squared method

%max number of iterations until the loop breaks
maxIters = 1000;

%initialize beta
beta = zeros(size(tX,2),1);

for k = 1:maxIters
    L = computeCost(y,tX,beta);
    g = computeGradient(y,tX,beta);
    
    %update beta
    beta = beta - alpha.*g;
    disp(g'*g);
    % stop if converged close enough
    if g'*g < 1e-5; break; end;
end

end


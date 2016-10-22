function [beta] = ridgeRegression(y, tX, lambda)
% Ridge regression using normal equations.
% 
% lambda is the regularization coefficient.

    % We put zero for the first beta in the regularization term.
    [~, M] = size(tX);
    lambdaMatrix = lambda*eye(M);
    lambdaMatrix(1,1) = 0;
    beta = (tX'*tX + lambdaMatrix)\(tX'*y);

end


function [beta] = leastSquaresGD(y, tX, alpha)
% Least squares using gradient descent
%
% alpha is the step-size

    % initialize
    [N,D] = size(tX);

    beta = zeros(length(tX(1,:)),1);
    maxIters = 1000;

    for k = 1:maxIters
        % Computing graident and updating step
        g = computeGradient(y, tX, beta);
        beta = beta - alpha.*g;

        % Convergence
        if g'*g < 1e-5; break;end;
    end
end


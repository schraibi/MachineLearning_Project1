function [beta] = logisticRegression(y, tX, alpha)
% Logistic regression using gradient descent or Newton's method.
%
% alpha is the step size, in case of gradient descent.
    
    [N,M] = size(tX);
    beta = zeros(M,1);
    maxIters = 1000;
    L = computeCost(y, tX, beta);

    for k = 1:maxIters
        % Calculating the negative of the gradient and the hessian
        sig = logisticFct(tX*beta);
        negGrad = 1/N.*tX'*(sig-y);
        hessian = 1/N.*tX'*diag(sig)*diag(1-sig)*tX;

        % Update beta
        dk = hessian \ negGrad;
        oldBeta = beta;
        beta = oldBeta - alpha.*dk;
        
        % Computing MSE
        oldL = L;
        L = computeCost(y, tX, beta);
        
        % Condition for convergence
        cond = negGrad'*negGrad;
        if cond < 1e-5 || (oldL-L) < 0
            beta = oldBeta;
            break;
        end
    end
end


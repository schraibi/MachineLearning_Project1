function [beta] = penLogisticRegression(y, tX, alpha, lambda)
% Logistic regression using gradient descent or Newton's method.
%
% alpha is the step size for gradient descent
% lambda is the regularization parameter
    
    [N,M] = size(tX);
    beta = zeros(M,1);
    maxIters = 1000;
    L = computeCost(y, tX, beta);
    
    % Doing iterations of the newton's method
    for i = 1:maxIters   
        
        % Calculating penalization terms by taking the Jacobian of the
        % regularization term for pen2 and its derivative for pen1
        pen1 = lambda*beta/N;
        pen1(1,1) = 0;
        pen2 = lambda*eye(M)/N;
        pen2(1,1) = 0;
        
        % Calculating the negative of the gradient and the hessian
        sig = logisticFct(tX*beta);
        negGrad = 1/N.*tX'*(sig-y)+pen1;
        hessian = 1/N.*tX'*diag(sig)*diag(1-sig)*tX+pen2;

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
            break;
        end
    end
end


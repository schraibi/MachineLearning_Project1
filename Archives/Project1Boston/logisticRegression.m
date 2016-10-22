function [ beta ] = logisticRegression( y, tX, alpha )
%LOGISTICREGRESSION Apply the logistic regression using gradient descent.


%max number of iterations for the loop
maxIters = 1000;
% initialize
beta = zeros(size(tX,2),1);
% compute initial cost
L = computeCostLL(y,tX,beta);

N = size(tX,1);
for k = 1:maxIters
    %compute sigma
    if (tX*beta<=0)
        zig = 1./(1+exp((-1)*tX*beta));
    else
        zig = exp(tX*beta)./(1+exp(tX*beta));
    end
    %compute the gradient
    g = (1/N)*tX'*(zig-y);
    
    
    %newton method (not used)
    %S = diag(zig)*diag(1-zig);
    %H = (1/N)*tX'*S*tX;
    %d = H\g;
    %save beta(k) and update to beta(k+1)
    oldbeta = beta;
    beta = beta - alpha.*g;
    
    %compute cost
    %L = computeCostLL(y,tX,beta);
    disp(num2str(g'*g));
    
    % check convergence
    if g'*g < 1e-5
        beta = oldbeta;
        break;
    end;
    
end

end


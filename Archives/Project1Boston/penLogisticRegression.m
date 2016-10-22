function [ beta ] = penLogisticRegression( y, tX, alpha, lambda )
%PENLOGISTICREGRESSION Apply the logistic regression using gradient descent
%and penalization to avoid overfiting.

%max number of iterations for the loop
maxIters = 1000;
% initialize
beta = zeros(size(tX,2),1);
% compute initial cost
L = computeCost(y,tX,beta);

N = size(tX,1);
b = size(beta,1);
for k = 1:maxIters
    if (tX*beta<=0)
        zig = 1./(1+exp((-1)*tX*beta));
    else
        zig = exp(tX*beta)./(1+exp(tX*beta));
    end
    %compute the gradient
    g = (1/N)*tX'*(zig-y);
    
    %Newton method (not used)
    %     S = diag(zig)*diag(1-zig)+ lambda * [zeros(N-b),zeros(N-b,b);zeros(b,N-b),diag(beta)^2];
    %     H = tX'*S*tX;
    %
    %     d = H\g; %calculate d s.t. H*d = g
    
    %update beta
    oldbeta = beta;
    M = 2*(beta);
    M(1,1) = 0;
    %save beta(k) and update to beta(k+1)
    beta = beta - alpha.*g + lambda* M;
    %compute cost and plot
    oldL = L;
    L = computeCostLL(y,tX,beta);
    %disp(num2str(g'*g));
    %check convergence
    if g'*g < 1e-5 
        beta = oldbeta;
        break;
    end;
end

end


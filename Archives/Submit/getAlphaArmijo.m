function [ alpha ] = getAlphaArmijo(y, tX, beta, g, p, cost )
%GETALPHAARMIJO Implements the linesearch method backtracking Armijo
%   tau and eta are armijo parameters
    alpha_init = 0.9; % 0.9
    tau = 0.5; % 0.5
    eta = 0.9;
    alpha = alpha_init;
    update = computeCostMSE(y, tX, beta+alpha*p);
    cond = update - cost - eta*alpha*g'*p;
    while cond > 0
        alpha = tau*alpha;
        update = computeCostMSE(y, tX, beta+alpha*p);
        cond = update - cost - eta*alpha*g'*p;
    end
end


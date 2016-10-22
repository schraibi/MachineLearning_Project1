function [beta, cost] = leastSquaresGD(y,tX,alpha_init)
%LEASTSQUARESGD Implements leastSquares through Gradient Descent
%   For ensured convergence (except if the function is unbounded), it
%   implements backtrackingArmijo to find the alpha
  % algorithm parameters
  maxIters = 1000;
  converged = 10^-4;
  alpha = alpha_init;

  % initialize
  beta = ones(size(tX,2),1);  

  % iterate
  for k = 1:maxIters
    % GRADIENT
    g = computeGradientMSE(y,tX,beta);

    % COST FUNCTION
    L = computeCostMSE(y, tX, beta);

    % GRADIENT DESCENT
    alpha = getAlphaArmijo(y, tX, beta, g, -g, L);
    %alpha = 0.1;
    beta = beta - alpha*g;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % CONVERGENCE CRITERIA
    if abs(alpha*g) < converged
        break;
    end
  end
  [cost, idxMin] = min(L_all);
  beta = beta_all(:,idxMin);
end
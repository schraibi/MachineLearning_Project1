function [ beta, cost ] = logisticRegressionPenalized( y, tX, alpha,lambda)
%LOGISTICREGRESSIONPENALIZED Implements the penalized version of logistic
%   It uses the newton step
    % algorithm parameters
  maxIters = 1000;
  converged = 10^-2;
  
  % initialize
  beta = zeros(size(tX,2),1); 

  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');
  fprintf('L  beta0 beta1\n');
  for k = 1:maxIters
    % Cost, Gradient, Hessian
    [L, g, H] = logisticRegLossPenalized(beta,y,tX, lambda);
    
    d = H\g; % Newton's Step
    beta = beta - alpha*d;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % CONVERGENCE CRITERIUM
    if abs(alpha*g) < converged
        break;
    end

  end
  [cost, idxMin] = min(L_all);
  beta = beta_all(:,idxMin);
  

end


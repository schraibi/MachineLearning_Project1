function [ beta, cost ] = logisticRegression( y, tX, alpha)
%LOGISTICREGRESSION Implements the logisticRegression algorithm
%   
    % algorithm parameters
  maxIters = 1000;
  converged = 10^-2;
  
  % initialize
  beta = zeros(size(tX,2),1); 

  % iterate
  fprintf('Starting iterations, press Ctrl+c to break\n');
  fprintf('L  beta0 beta1\n');
  for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    [L, g, H] = logisticRegLoss(beta,y,tX);
    
    d = H\g; % Newton's Step
    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta - alpha*d;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % INSERT CODE FOR CONVERGENCE
    if abs(alpha*g) < converged
        break;
    end

  end
  [cost, idxMin] = min(L_all);
  beta = beta_all(:,idxMin);
  

end


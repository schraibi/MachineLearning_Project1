import numpy as np

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_*np.dot(np.transpose(w),w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w) + 2*lambda_*sum(w)
    return loss, gradient, hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    
    w = w - gamma*np.dot(np.linalg.inv(hessian),gradient)
    
    return loss, w
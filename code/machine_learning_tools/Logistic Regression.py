import numpy as np

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss  = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian


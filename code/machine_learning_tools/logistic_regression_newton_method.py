import numpy as np

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w = w - gamma*np.dot(np.linalg.inv(hessian),gradient)
    
    return loss, w

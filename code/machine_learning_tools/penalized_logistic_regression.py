import numpy as np
from machine_learning_tools.sigmoid_loss_gradient_and_hessian_for_logistic_regression import *

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

def penalized_logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
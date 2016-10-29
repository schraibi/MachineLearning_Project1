from machine_learning_tools.gradient_descent import gradient_descent
from machine_learning_tools.stochastic_gradient_descent import stochastic_gradient_descent
from machine_learning_tools.least_squares import least_squares
from machine_learning_tools.ridge_regression import ridge_regression
from machine_learning_tools.penalized_logistic_regression import penalized_logistic_regression_gradient_descent


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 4
    return def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):

def least_squares(y, tx):
    return least_squares.least_squares(y, tx)

def ridge_regression(y, tx, lambda_):
    return ridge_regression.ridge_regression(y, tx, lambda_)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return penalized_logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma, 0.0)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return penalized_logistic_regression_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_)

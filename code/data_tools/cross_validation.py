import numpy as np
from util.helpers import build_poly
from machine_learning_tools.least_squares import *
from machine_learning_tools.costs import *

def cross_validation(y, x, k_indices, k, degree):
    test_y = y[k_indices[k]]
    test_x = x[k_indices[k]]
    k_indices =np.delete(k_indices,k,0)
    train_y = y[k_indices.flatten()]
    train_x = x[k_indices.flatten()]
    
    fi_train_x=build_poly(train_x,degree)
    fi_test_x=build_poly(test_x,degree)
    
    w = least_squares(train_y,fi_train_x)
    loss_tr = np.sqrt(2*calculate_mse(train_y-fi_train_x.dot(w)))
    loss_te = np.sqrt(2*calculate_mse(test_y-fi_test_x.dot(w)))
    
    return loss_tr, loss_te
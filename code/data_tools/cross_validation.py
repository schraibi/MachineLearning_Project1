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

def cross_validation_demo():
    seed = 6
    degree = 30
    k_fold = 4
    #lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    rmse_tr_lambda=0
    rmse_te_lambda=0
    for k in range(k_fold):
        rmse_tr_k, rmse_te_k = cross_validation(y,tX,k_indices,k,degree)
        rmse_tr_lambda+=rmse_tr_k
        rmse_te_lambda+=rmse_te_k
    print(rmse_tr_lambda/k_fold)
    print(rmse_te_lambda/k_fold)
    #rmse_tr.append(rmse_tr_lambda/k_fold) 
    #rmse_te.append(rmse_te_lambda/k_fold)    
    
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)
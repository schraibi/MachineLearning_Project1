# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
from machine_learning_tools.costs import *

def compute_gradient(y, tx, w):
  e = y - np.dot(tx, w)
  gradient = np.dot(np.transpose(tx), e) / (-len(y))
  return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
  # Define parameters to store w and loss
  ws = [initial_w]
  losses = []
  w = initial_w
  for n_iter in range(max_iters):
    # compute gradient and loss
    loss = compute_loss_mse(y, tx, w)
    gradient = compute_gradient(y, tx, w)
    # update w by gradient
    w = w - gamma*gradient
    # store w and loss
    ws.append(np.copy(w))
    losses.append(loss)
    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

  return losses, ws

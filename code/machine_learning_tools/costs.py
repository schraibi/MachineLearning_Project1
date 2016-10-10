# -*- coding: utf-8 -*-
"""functions used to compute the loss."""

import numpy as np

def compute_loss_mse(y, tx, w):
  e = y - np.dot(tx, w)
  mse = np.sum(np.square(e)) / (2*len(y))
  return mse

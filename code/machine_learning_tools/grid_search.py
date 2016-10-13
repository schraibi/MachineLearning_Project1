# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Grid Search
"""

import numpy as np
from machine_learning_tools.costs import *


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    losses = np.array([])
    for w0_elem in w0:
        for w1_elem in w1:
            losses = np.append(losses, compute_loss(y, tx, [w0_elem, w1_elem]))
    losses = losses.reshape((len(w0),len(w1)))
    return losses

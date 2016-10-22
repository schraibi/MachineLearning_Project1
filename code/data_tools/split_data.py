# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.shuffle(y)
    first_set_length = int(len(y)*ratio)
    second_set_length = len(y) - first_set_length
    first_y = y[0:first_set_length]
    second_y = y[first_set_length:len(y)]
    first_x = x[0:first_set_length]
    second_x = x[first_set_length:len(y)]
    return ((first_y, first_x),(second_y, second_x))



x = np.array([
    [2, 5, 3, 2.5 ],
    [7, 1, 2, 1   ],
    [0, 1, 3, 0.2 ]])

y = [9,
     8,
     7]

k_indices = [0,2]

x2 = np.array([x[i,:] for i in k_indices])
print(x2)
print(split_data(x,y,0.4))
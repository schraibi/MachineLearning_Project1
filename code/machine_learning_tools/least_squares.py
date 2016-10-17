import numpy as np

def least_squares(y, tX):
    """calculate the least squares solution."""
    """ovde moze da bude problem ako je matrica ill conditioned!!! proveri!!!"""
    XXT = np.transpose(tX).dot(tX)
    XXTI = np.linalg.inv(XXT)
    return XXTI.dot(np.transpose(tX)).dot(y)
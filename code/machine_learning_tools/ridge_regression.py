import numpy as np

def ridge_regression(y, tX, lambda_):
    """implement ridge regression."""
    XXT = np.transpose(tX).dot(tX)
    XXT_ridge = XXT + 2*len(y)*lambda_*np.identity(tX.shape[1])
    XXTI = np.linalg.inv(XXT_ridge)
    XXTIT = XXTI.dot(np.transpose(tX))
    return XXTIT.dot(y)
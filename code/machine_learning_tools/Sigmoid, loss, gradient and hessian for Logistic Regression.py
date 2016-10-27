import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return (np.exp(t))/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    for i in range(0,len(y)):
     loss +=  np.log(1 + np.exp(np.transpose(tx).dot(w))) - y[i]*np.transpose(tx).dot(w)
        
    return  sum(loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(np.transpose(tx),np.apply_along_axis(sigmoid,axis=1,arr=np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    
    gradient = calculate_gradient(y, tx, w)
    
    w = w - np.multiply(gamma,gradient)
    
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    """"sig = np.apply_along_axis(sigmoid,axis=1,arr=np.dot(tx,w))
    S = np.diag(np.diag(np.dot(sig,np.ones((len(y),1))-sig)))"""
    S = np.zeros((len(y),len(y)))
    for i in range(len(y)):
        S[i][i] = np.dot(sigmoid(np.dot(tx[i],w)), 1 - sigmoid(np.dot(tx[i],w)))

    return np.dot(np.transpose(tx),np.dot(S,tx))


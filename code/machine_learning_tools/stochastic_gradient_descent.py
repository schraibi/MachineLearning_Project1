# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from util.helpers import batch_iter

def compute_stoch_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    gradient = np.dot(np.transpose(tx), e) / (-len(y))
    return gradient

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    current_epoch=0
    while current_epoch < max_epochs:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            if(current_epoch >= max_epochs):
                break
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)
            current_epoch = current_epoch + 1
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=current_epoch, ti=max_epochs - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

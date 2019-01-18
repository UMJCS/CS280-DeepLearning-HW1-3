import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(len(X)):
        index = y[i]
        scores = X[i].dot(W)
        scores = scores - np.max(scores)
        correct_class_score = scores[index]
        loss -= np.log(np.exp(scores[index])/(np.sum(np.exp(scores))))
        for j in xrange(num_classes):
            wrong = np.exp(scores[j])/np.sum(np.exp(scores)) + np.max(scores)
            right = np.exp(scores[index])/np.sum(np.exp(scores)) + np.max(scores)
            if j==y[i]:
                dW[:,j] = dW[:,j] + (right-1) * X[i]
            else:
                dW[:,j] = dW[:,j] + wrong * X[i]


    loss = (loss/float(len(X))) + reg * np.sum(np.square(W))/2 + loss
    
    dW = dW / len(X) +  reg * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    
    
    loss = 0.0
    dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    num_classes = W.shape[0]
    num_train = X.shape[0]
    i_ran = np.arange(len(X))
    
    scores = np.dot(X,W)
    scores-= np.max(scores)
    cor = scores[i_ran,y].reshape(len(X),1)

    
    exp_sum = np.sum(np.exp(scores),axis=1).reshape(num_train,1)
    loss = loss + np.sum(np.log(exp_sum) - cor)
    loss = (loss/float(len(X))) + reg * np.sum(np.square(W))/2 + loss
    
    
    Grad = np.exp(scores)/exp_sum
    Grad[i_ran,y] = Grad[i_ran,y]-1.0
    dW = np.dot(X.T,Grad)/num_train + reg*W



    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


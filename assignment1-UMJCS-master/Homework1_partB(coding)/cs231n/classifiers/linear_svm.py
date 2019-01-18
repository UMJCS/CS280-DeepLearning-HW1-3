import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    
    dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        index = y[i]
        scores = X[i].dot(W)
        correct_class_score = scores[index]
        for j in xrange(num_classes):
          if j == y[i]:
            continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            loss += margin            
            dW[:,j]+=X[i]
            dW[:,index]-=X[i]
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    dW = dW / len(X) + 2 *reg * W  
    loss = loss/len(X)

    # Add regularization to the loss.

    loss += 0.5 * reg * 2* np.sum(W * W) 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    
    num_classes = W.shape[0]
    num_train = X.shape[0]
    i_ran = np.arange(len(X))
    
    scores = np.dot(X,W)

    cor = scores[i_ran,y].reshape(len(X),1)

    margin = scores - cor + 1
    margin[margin < 0] = 0
    margin[i_ran,y] = 0
    loss = (margin.sum()/len(X)) + reg * np.sum(np.square(W))/2 + loss
    

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

    margin[margin > 0] = 1
    row_sum = np.sum(margin,axis=1)
    margin[np.arange(num_train),y] = -row_sum
    dW = np.dot(X.T,margin)/num_train + reg*W
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return loss, dW

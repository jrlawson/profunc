# -*- coding: utf-8 -*-
__docformat__= 'restructuredtext en'

import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    """Multi-class logistic regression class
    
    The logistic regression is fully described by a weight matrix :math:'W'
    and bias vector :math:'b'. Classification is done by projecting data points
    onto a set of hyperplanes, the distance to which is used to determine a 
    class membership probability.
    """
    
    def __init__(self, input, n_in, n_out):
        """Initialize the parameters of the logistic regression
        
        :type input:    theano.tensor.TensorType
        :param input:   symbolic variable that describes the input of the 
                        architecture (one minibatch)
                        
        :type n_in:     int
        :param n_in:    number of input units, the dimension of the space in
                        which the datapoints lie

        :type n_out:    int
        :param n_out:   number of output units, the dimension of the space in 
                        which the labels lie
        """
        # Initialize to 0 the weights W as a (n_in x n_out) matrix.
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in,n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )

        # Initialize the biases b as an n_out length vector of 0s.
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out, ),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        # Symbolic expression for computing the matrix of class membership 
        # probabilities where:
        #   W   is a matrix with column-k representing the separation 
        #       hyperplane for  class-k
        #
        #   x   is a matrix with row-j representing input training sample j
        #        
        #   b   is a vector with element k representing the free parameter of 
        #       hyperplane k.
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Symbolic description of how to compute prediction as a class with 
        # maximal probability. The axis argument is the axis along which to
        # compute the index of the argmax.
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        # Parameters of the model
        self.params = [self.W, self.b]
        
        
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        
        (math stuff)
        
        Note:   We use the mean instead of the sum so that the learning rate
                is less dependent on the batch size.
        """
        #   *   y.shape[0] : Symbolically, the number of rows in y, i.e., the  
        #       number of examples (call that n) in the minibatch. 
        #
        #   *   T.arange(y.shape[0] is a symbolic vector that will contain 
        #       [0,1,..,n-1].
        # 
        #   *   T.log(self.p_y_given_x) is a matrix of log-probabilities 
        #       (call it LP) with one row per example and one column per class.
        #
        #   *   LP[T.arange(y.shape[0]), y] is a vector, call it v, containing 
        #       [LP[0,y[0]], [LP[1,y[1]], LP[2,y[2]],..,[LP[n-1,y[n-1]]] 
        #        
        #   *   T.mean(LP[T.arange(y.shape[0]),y]) is the mean across minibatch 
        #       examples of the elements in v, i.e. the mean log-likelihood 
        #       across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        divided by the total number of examples in the minibatch ; zero one
        loss over the size of the minibatch (???)
        
        :type y:    theano.tensor.TensorType
        :param y:   corresponds to a vector that gives the correct label for
                    each example
        """

        # Check to insure that y has the same dimension as y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check to insure that y is of the correct data type
        if y.dtype.startswith('int'):
            # The T.neq operator returns a vector of 0s and 1s where 1 
            # represents a mistake in prediction.
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()                    
    

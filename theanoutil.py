# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:31:33 2015

@author: jlawson
"""
import theano
import theano.tensor as T
import numpy

# train_set, valid_set, test_set format: tuple(index, target)
    #
    # input:    A 2-D numpy.ndarray (a matrix) where each row corresponds
    #           to an example. 
    #
    # target:   A 1-D numpy.ndarray (a vector) of length equal to 
    #           input.rows. It should give the target to the example with
    #           same index as the input.
    
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    
    data_xy:type:   An array of array of floatX corresponding to an
                    example or observation. The first element of the array
                    is the input matrix and the second element is the 
                    output vector.
    
    Returns an array of pairs of the form:
    
    [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    All of these elements are Theano shared variables. The reason we store 
    the dataset in shared variables is to allow Theano to copy it to
    GPU memory (when configured to run on a GPU).
    
    Since copying data to GPU is slow, it is necessary to copy a 
    minibatch at a time. The default behavior if the dataset variables
    are not shared would lead to substantial decline in performance.
    
    If you are using a GPU and the GPU does not have sufficientt memory to
    store all of the dataset, you will probably get an error. Currently,
    the work around is to not use the GPU. However, we probably need an
    alternative method to move data to the GPU a minibatch at a time.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
                                 
    # When storing data on the GPU it needs to be as floats (Float32).
    # Therefore we store the labels as ''floatX'' as well. That is 
    # shared_y's job. But during computation we need the labels as ints
    # because we use them as indices. So instead of returning shared_y
    # directly, we cast it to an int.
    return shared_x, T.cast(shared_y, 'int32')
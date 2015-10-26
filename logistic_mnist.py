# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:47:47 2015

@author: jlawson
"""

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import pickle_data_loader
from logistic_sgd import LogisticRegression

def sgd_optimization(datasets, learning_rate=0.13, n_epochs=1000, datasets, batch_size=600, image_width=28, image_height=28, num_classes=10):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model using the MNIST data.
    
    :type datasets: Shared array of array of floats. Three elements in the top
                    level array, two elements in each subarray.
    :param datasets: The three top level elements are the training set, the
                    validation set, and the test set, respectively. Each of 
                    these is a two element set containing the inputs (a matrix
                    where each row is an example) and the outputs/labels, which
                    is a vector for binary classifiers and a matrix otherwise,
                    and each "row" is the output. 
        
    :type learning_rate:    float
    :param learning_rate:   Learning rate used for stochastic gradient
                            descent algorithm
              
    :type n_epochs:         int
    :param n_epochs:        Maximum number of epochs to run the optimizer
    
    :type batch_size:       int
    :param batch_size:      The size of the minibatches
    """

    print '... Loading data'    
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]
    
    # Compute number of minibatches for training, validation, and testing
    # A "borrow" option is available for shared variables. It is False by 
    # When false, the shared variable we construct gets a deep copy of the 
    # data. So changes we subsequently make to the data have no effect on the
    # shared variable. When borrow is True, as in our case, we will do a 
    # shallow copy when sharing. This is faster, but we need to be careful
    # to make sure that it is immutable.
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ####################
    # Build actual model
    ####################
    print '... Building the model'
    
    # Allocate symbolic variables for the data
    index = T.lscalar()     # index to a minibatch
    
    # Generate symbolic variables for input. The x and y represent a minibatch
    x = T.matrix('x')       # Rasterized image data
    y = T.ivector('y')      # Data labels 
    
    # Construct an instance of the logistic regression class. The number of 
    # inputs is equal to the number of pixels in the images, and the number of
    # outputs is equal to the number of classes.
    classifier = LogisticRegression(input=x, n_in=image_width * image_height, n_out=num_classes)

    # Define the cost function that we will minimize during training. That 
    # function is the negative log-likelihood of the model in symbolic 
    # format.
    cost = classifier.negative_log_likelihood(y)

    # Compile a Theano function that computes the mistakes made on a minibatch
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            # Take the appropriate slices of the arrays to define the batch
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            # Take the appropriate slices of the arrays to define the batch
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # Compute the gradients
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # Specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
           
    # Compiling a Theano function 'train_model' that returns the cost, but at
    # the same time updates the parameter of the model based on the rules
    # defined in 'updates'.
    # 
    # Each time train_model(index) is called, it will compute and return the 
    # cost of a minibatch, while also performing a step of MSGD. The entire
    # learning algorithm consists of looping over all examples in the data set,
    # considering all examples in one minibatch at a time, and repeatedly
    # calling train_model(index).
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # Train model
    ###############
    print '... Training the model'
    # Earliy-stopping parameters
    patience = 5000                  # Examine at least this many examples regardless
    patience_increase = 2            # Wait at least this much longer when a new best is found
    improvement_threshold = 0.995    # A relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)   # Go through this many minibatches before checking the network on the validation set. In this case, we check every epoch
    
    best_validation_loss = numpy.inf # Rather meaningless initializations.
    test_score = 0
    start_time = time.clock()
    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # Compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)                        
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )                        
                )
                
                # If we got the best validation score thus far...
                if this_validation_loss < best_validation_loss:
                    # Improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    
                    # Test it on the test set
                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
           
                    print(
                        (
                            '    epoch %i, minibatch %i/%i, test error of best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
            if patience <= iter:
                done_looping = True
                break
            
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code ran for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for the file ' + os.path.split(__file__)[1] + ' ran for %.lfs' % ((end_time - start_time)))
        
if __name__ =='__main__':
    datasets = pickle_data_loader.PickleDataLoader().load_data('mnist.pkl.gz')  # Everything comes back shared.'mnist.pkl.gz'
    sgd_optimization()
    
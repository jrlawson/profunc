# -*- coding: utf-8 -*-huj
"""
Created on Thu Mar 19 14:56:56 2015

@author: jlawson
"""
import os
import gzip
import cPickle
import theanoutil

def load_image_data(dataset):
    ''' Loads the dataset with the path given by the dataset parameter. If no
    such dataset is available locally, loads the MNIST dataset over the 
    network.
    
    The data is assumed to be in pickled form with three elements:
    
    1) Training
    2) Validation
    3) Test
        
    :type dataset:  string
    :param dataset: the path to the dataset
    '''
    ###############
    # LOAD DATA
    ###############
        
    # If dataset is not present locally, download MNIST data from the network. 
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
                
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
        
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
        
  
            
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
        
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval
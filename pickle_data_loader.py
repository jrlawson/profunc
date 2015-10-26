# -*- coding: utf-8 -*-huj
"""
Created on Thu Mar 19 14:56:56 2015

@author: jlawson
"""
import os
import gzip
import cPickle
import theanoutil
import data_loader

class PickleDataLoader(data_loader.DataLoader):
    '''A concrete DataLoader that loads data from a pickle file.
    '''
    def __init__(self):
        self.debug = True
        
    def load_data(self, dataset):
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
        training_set, validation_set, test_set = cPickle.load(f)
        f.close()
                        
        test_set_x, test_set_y = theanoutil.shared_dataset(test_set)
        validation_set_x, validation_set_y = theanoutil.shared_dataset(validation_set)
        training_set_x, training_set_y = theanoutil.shared_dataset(training_set)

        if self.debug:         
            print "TYPE of test_set_x =", type(test_set_x)        
            print "TYPE of test_set=", type(test_set), "  SIZE of test_set=", len(test_set)
            print "TYPE of test_set[0]=", type(test_set[0]), "  SHAPE of test_set[0]=", test_set[0].shape
            print "TYPE of test_set[1]=", type(test_set[1]), "  SHAPE of test_set[1]=", test_set[1].shape
            print "VALUE of training_set[0,0,0]=", training_set[0][0,0]
            print "VALUE of training_set[1,0]=", training_set[1][0]
            
        rval = [(training_set_x, training_set_y), (validation_set_x, validation_set_y), (test_set_x, test_set_y)]
        return rval
        
if __name__ =='__main__':
    PickleDataLoader().load_data('mnist.pkl.gz')
    
    
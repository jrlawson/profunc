# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:51:41 2015

@author: jlawson
"""
from abc import ABCMeta, abstractmethod

class DataLoader:
    '''An abstract base class for loading data into a Theano deep learning 
    application.
    '''    
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_data(self, source):
        '''Abstract method to load the data from a source. Returns an array of
        pairs in the form:
        
        [(train_set_in, train_set_out), (validation_set_in, validation_set_out), (test_set_in, test_set_out)]

        :type source:   arbitrary
        :param source:  Where to find the data. It could be a URL, or a file, 
        or a database query, etc.
        '''
        pass
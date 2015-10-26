# -*- coding: utf-8 -*-
__docformat__= 'restructuredtext en'

from stochastic_gradient_descent import sgd_optimization
from pickle_data_loader import PickleDataLoader

if __name__ =='__main__':    
    datasets = PickleDataLoader().load_data('mnist.pkl.gz')    
    sgd_optimization(datasets)
    
# -*- coding: utf-8 -*-
__docformat__= 'restructuredtext en'

from stochastic_gradient_descent import sgd_optimization
from text_file_data_loader import TextFileDataLoader

if __name__ =='__main__':    
    datasets = TextFileDataLoader().load_data("dipep_only_others")   
    sgd_optimization(datasets, image_width=20, image_height=20, num_classes=11)

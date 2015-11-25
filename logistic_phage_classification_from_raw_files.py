# -*- coding: utf-8 -*-
__docformat__= 'restructuredtext en'

from stochastic_gradient_descent import sgd_optimization
from multi_reader import MultiReader

if __name__ =='__main__':    
    datasets = MultiReader().load_data("/home/jlawson/Dropbox/ProteinFunctionData/")   
    sgd_optimization(datasets, image_width=20, image_height=20, num_classes=11)

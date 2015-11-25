# -*- coding: utf-8 -*-
__docformat__= 'restructuredtext en'


from multi_layer_perceptron import run_mlp
import math
from text_file_data_loader import TextFileDataLoader
from multi_reader import MultiReader
from principled_multi_loader import PrincipledMultiLoader

class MyDataLoader(TextFileDataLoader):
    def preprocess(self, data):
        n = self.input_width
        num_examples = len(data)
        for c in range(0,n):
            sum_x = 0.0
            sum_x2 = 0.0
            for r in range(0, num_examples):                
                sum_x += data[r,c]
                sum_x2 += data[r,c]*data[r,c]
            mu = sum_x / num_examples
            variance = (sum_x2 - (sum_x*sum_x)/num_examples)/num_examples
            std = math.sqrt(variance)  # Population std
            for r in range (0, num_examples):
                z = (data[r,c] - mu) / std
                data[r,c] = z
            if c % 10 == 0:
                print "Preprocessed column ", c
        return data

# Using TextDataLoader and run_mlp(datasets, n_hidden=800, image_width=20, image_height=20, num_classes=11, learning_rate=0.005)
# we get a test error rate of about 22.3%.

if __name__ =='__main__':    
    datasets = PrincipledMultiLoader().load_data("/home/jlawson/Dropbox/ProteinFunctionData/") 
    #datasets = MultiReader().load_data("/home/jlawson/Dropbox/ProteinFunctionData/")
    run_mlp(datasets, n_hidden=400, image_width=20, image_height=20, num_classes=11, learning_rate=0.00125)
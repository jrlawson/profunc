# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:32:26 2015

@author: jlawson
"""

import math
from multi_reader import MultiReader

class PrincipledMultiLoader(MultiReader):
    def __init__(self, output_width=11, training_frac=70.0, validation_frac=15.0, squashed=True, debug=False):
        MultiReader.__init__(self, output_width=11, training_frac=70.0, validation_frac=15.0, debug=False)
        self.squashed = squashed
        
    def preprocess(self, data):
        cols = self.input_width
        rows = len(data)
        for c in range(0, cols):        
            sum_x = 0.0
            sum_x2 = 0.0
            for r in range(0, rows):
                sum_x += data[r,c]
                sum_x2 += data[r,c] * data[r,c]
            mu = sum_x / rows
            std = math.sqrt((sum_x2 - (sum_x*sum_x)/rows)/rows)  # Population std

            for r in range (0, rows):
                z = (data[r,c] - mu) / std
                if self.squashed:
                    z = sigma(z)
                data[r,c] = z
            if c % 10 == 0:
                print "Preprocessed column ", c
        return data
        

def sigma(x):
    return 1 / (1 + math.exp(-x))
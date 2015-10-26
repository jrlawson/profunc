# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:12:29 2015

@author: jlawson
"""
import math
from random import randint, seed
import numpy
from data_loader import DataLoader
import theanoutil

class TextFileDataLoader(DataLoader):
    def __init__(self, input_width=400, output_width=11, training_frac=70.0, validation_frac=15.0, debug=False):
        self.input_width = input_width
        self.output_width = output_width
        self.training_frac = training_frac
        self.validation_frac = validation_frac
        self.debug = debug
        
        
    def load_data(self, source):
        f = file(source, "r")
        string_data = [line.split() for line in f]
        f.close()
        
        num_examples = len(string_data)
        if self.debug:
            print len(string_data)
            print len(string_data[1])
            print string_data[0][0]
        
        # But the labeled data we have is not randomly ordered. It is sorted
        # by class. We need to shuffle it up or we will only train on the first
        # classes.
        shuffle = self.rand_perm(num_examples)

        data = numpy.ndarray((num_examples, self.input_width+self.output_width), float)
        for n in range(0, num_examples):
            for w in range(0,self.input_width+self.output_width):
                s = string_data[shuffle[n]][w]
                data[n,w] = float(s)

        data = self.preprocess(self.cull(data))
        num_examples = len(data)
        print "Data shape = ", data.shape, "   num_examples=", num_examples
        inputs = numpy.array(data)[:, 0:self.input_width]
        outputs_full = numpy.array(data)[:, self.input_width:self.input_width+self.output_width]
        
        if self.debug:
            print inputs.shape
            print outputs_full.shape
        outputs = numpy.ndarray((num_examples,),int)       
        for n in range(0, num_examples):
            found_class = False
            for w in range(0, self.output_width):
                if outputs_full[n,w] > 0.5:
                    outputs[n] = w
                    found_class = True
                    break
        num_training_cases = self.num_training(num_examples)
        num_validation_cases = self.num_validation(num_examples)
        num_test_cases = self.num_test(num_examples)
        
        print num_training_cases, " ", num_validation_cases, " ", num_test_cases
        training_set = (inputs[0:num_training_cases,:], outputs[0:num_training_cases])
        validation_set = (inputs[num_training_cases:num_training_cases+num_validation_cases,:], outputs[num_training_cases:num_training_cases+num_validation_cases])
        test_set = (inputs[num_training_cases+num_validation_cases:,:], outputs[num_training_cases+num_validation_cases:])
        training_set_x, training_set_y = theanoutil.shared_dataset(training_set)
        validation_set_x, validation_set_y = theanoutil.shared_dataset(validation_set)
        test_set_x, test_set_y = theanoutil.shared_dataset(test_set)
        
        if self.debug:
            print "TYPE of test_set_x =", type(test_set_x)        
            print "TYPE of test_set=", type(test_set), "  SIZE of test_set=", len(test_set)
            print "TYPE of test_set[0]=", type(test_set[0]), "  SHAPE of test_set[0]=", test_set[0].shape        
            print "TYPE of test_set[1]=", type(test_set[1]), "  SHAPE of test_set[1]=", test_set[1].shape
            print "VALUE of training_set[0,0,0]=", training_set[0][0,0]
            print "VALUE of training_set[1,0]=", training_set[1][0], "   test_set[1,0]=",test_set[1][0]
        
        rval = [(training_set_x, training_set_y), (validation_set_x, validation_set_y), (test_set_x, test_set_y)]
        return rval
     
    def num_training(self, num_examples):
        return num_examples * (self.training_frac/100.0)
    
    def num_validation(self, num_examples):
        return num_examples * (self.validation_frac/100.0)
        
    def num_test(self, num_examples):
        return num_examples - (self.num_training(num_examples) + self.num_validation(num_examples))
        
    def rand_perm(self, length):
        # In debug mode, we want to have a repeatable random number seed so
        # that we can have a repeatable shuffling.
        if self.debug:
            seed(1)
        shuffle = numpy.ndarray((length,), int)
        for n in range(0, length):
            shuffle[n]=n
        for n in range(0, length):
            swap_cell = randint(0, length-1)
            temp = shuffle[swap_cell]
            shuffle[swap_cell] = shuffle[n]
            shuffle[n] = temp            
        return shuffle
        
    def cull(self, data):
        # Make a list of all row numbers that need to get culled from the data.
        cull_list = []
        for n in range(0, len(data)):
            if self.prune(data[n]):
                cull_list.append(n)
        cull_list.append(len(data))  # A sentinel at the end of the cull list.
        
        # Make a new array that doesn't have the culled items in it.
        # The 1+ is for the sentinel.
        new_data = numpy.ndarray((1+len(data)-len(cull_list), self.input_width+self.output_width), float)
        next_cull_index = 0
        next_data_index = 0
        for n in range(0, len(data)):
            if n == cull_list[next_cull_index]:
                next_cull_index += 1
            else:
                new_data[next_data_index] = data[n]
                next_data_index += 1
        print "Number culled = ", len(cull_list)-1                
        return new_data
            
    def prune(self, example):
        sum = 0.0
        for n in range(0, self.input_width):
            if example[n] < 0.0:
                return True
            if example[n] > 1.0:
                return True
            sum += example[n]
        if sum > 1.01:
            return True
        if sum < 0.99:
           return True
        return False

        
    def preprocess(self, data):
        n = self.input_width
        for r in range(0, len(data)):
            sum_x = 0.0
            sum_x2 = 0.0
            for c in range(0, n):
                sum_x += data[r,c]
                sum_x2 += data[r,c]*data[r,c]
            mu = sum_x / n
            std = math.sqrt((sum_x2 - (sum_x*sum_x)/n)/n)  # Population std
            for c in range (0, n):
                z = (data[r,c] - mu) / std
                #squashed_z = sigma(z)
                data[r,c] = z
            if r % 1000 == 0:
                print "Preprocessed row ", r
        return data
        
if __name__ =='__main__':
    TextFileDataLoader(debug=True).load_data("dipep_only_others")

from Bio import SeqIO
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
import math
import numpy
from random import randint, seed
import theanoutil



class MultiReader(DataLoader):
    def __init__(self, output_width=11, training_frac=70.0, validation_frac=15.0, debug=False):
        self.input_width = 400
        self.output_width = output_width
        self.training_frac = training_frac
        self.validation_frac = validation_frac
        self.debug = debug
        #self.dir = "/home/jlawson/Dropbox/ProteinFunctionData/"      # Where the files live.
        self.names =[                                                # Names of all of the files.
            "baseplate_3370", 
            "collar_1385", 
            "htj_2258_nofg", 
            "major_tail_1512", 
            "mcp_3589", 
            "minor_capsid_1500_nofg", 
            "minor_tail_2033", 
            "portal_2141",
            "tail_fiber_3007",
            "tail_sheath_2350"]

        self.feature_extractor = FeatureExtractor()

    def load_data(self, source):
        '''Load the data from a directory with a collection of source files,
        one file for each kind of protein. 
        
        Returns an array of pairs in the form:
        
        [(train_set_in, train_set_out), (validation_set_in, validation_set_out), (test_set_in, test_set_out)]

        :type source:   String
        :param source:  The directory where the source files are located.
        '''
        dir = source        
        raw_data = list()
        unsupporteds = list()            
        for i in range(0,len(self.names)) :
            num_in_file = 0
            if self.debug:
                print(dir + self.names[i] + ".faa")
            handle = open(dir+self.names[i]+".faa", "rU")            # Open a file.
            for record in SeqIO.parse(handle, "fasta") :
                num_in_file+=1
                try:
                    #print "      " + record.id
                    feature_vector = self.feature_extractor.extract_features(record)
                    # Now we have to augment the feature vector with the output
                    # vector. So we:
                    #   1) Make a new array a bit longer than the feature vector, 
                    #   2) Copy the feature vector into the first cells of the new array,
                    #   3) Find the appropriate cell in the tail of the new array
                    #      and set that one equal to 1.
                    prepared_data_record = numpy.zeros(len(feature_vector) + self.output_width)
                    for col in range(0,len(feature_vector)):                        # This surely could be done more efficiently.
                        prepared_data_record[col] = feature_vector[col]             # Doesn't matter for now.
                    prepared_data_record[len(feature_vector)+i] = 1                 # The class of the protein is taken from the order of the files in the list "names"
                    raw_data.append(prepared_data_record)
                except KeyError:
                    if self.debug:
                        print "   Unsupported sequence: " + record.id + "   " + str(record.annotations)
                    unsupporteds.append(record)
                pass
            handle.close()
            if self.debug:
                print "Total in file " + self.names[i] + " = " + str(num_in_file)
        
        # Now we are done reading all of the data in. In debug mode, print some
        # overall summary information.
        if self.debug:
            print "Supported Sequences = " + str(len(raw_data))
            print "Unsupported Sequences = " + str(len(unsupporteds))
        
        
        num_examples = len(raw_data)
        
        # But the labeled data we have is not randomly ordered. It is sorted
        # by class. We need to shuffle it up or we will only train on the first
        # classes.
        if self.debug:
            print "Shuffling data to randomize for training"
        shuffle = self.rand_perm(num_examples)

        data = numpy.ndarray((num_examples, self.input_width+self.output_width), float)
        for n in range(0, num_examples):
            for w in range(0,self.input_width+self.output_width):
                s = raw_data[shuffle[n]][w]
                data[n,w] = float(s)                
        if self.debug:
            print "Finished shuffling data"
            print "Processing data to cull outliers"
        data = self.preprocess(self.cull(data))
        num_examples = len(data)
        print "Data shape = ", data.shape, "   num_examples=", num_examples
        inputs = numpy.array(data)[:, 0:self.input_width]
        outputs_full = numpy.array(data)[:, self.input_width:self.input_width+self.output_width]
        if self.debug:
            print "Finished culling outliers"
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
     

 
 
 
 
 
 
    # Everything from here down should be turned into a base class. 
 
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
    reader = MultiReader(debug=True)
    data = reader.load_data("/home/jlawson/Dropbox/ProteinFunctionData/")           
#for i in range(0,len(raw_data)) :
#    print(raw_data[i].id)

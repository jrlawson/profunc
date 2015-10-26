# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:31:12 2015

@author: jlawson
"""


from stacked_denoising_autoencoder import run_SdA
from principled_text_file_data_loader import PrincipledTextFileDataLoader

# run_sdA defaults results in 22.17% test error


#run_sda_phage(hidden_layer_sizes=[1000, 1000, 1000, 1000, 1000], corruption_levels=[0.25, 0.25, 0.25, 0.25, 0.25], pretraining_epochs=25, batch_size=5)
# 22.17% test error (but runs for 400 epochs)

# run_sda_phage(hidden_layer_sizes=[1000, 800, 600], corruption_levels=[0.1, 0.2, 0.4], pretraining_epochs=25, batch_size=5):
# 21.79% test error

# hidden_layer_sizes=[1000, 1000, 800, 800], corruption_levels=[0.1, 0.2, 0.3, 0.4], pretraining_epochs=20, batch_size=2
# 22.04% test error

# hidden_layer_sizes=[1000, 1000, 1000], corruption_levels=[0.1, 0.2, 0.4], pretraining_epochs=25, batch_size=5
# 22.83% test error

# hidden_layer_sizes=[800, 400, 200], corruption_levels=[0.1, 0.2, 0.3], pretraining_epochs=25, batch_size=5):
# 23.17% test error

# hidden_layer_sizes=[1600, 1200, 800], corruption_levels=[0.2, 0.3, 0.4], pretraining_epochs=25, batch_size=3
# 24.1% test error

# hidden_layer_sizes=[100, 100], corruption_levels=[0.3, 0.3], pretraining_epochs=15, batch_size=5
# 26.65% test error

# hidden_layer_sizes=[300, 100], corruption_levels=[0.4, 0.2], pretraining_epochs=15, batch_size=5
# 27.45% test error

# hidden_layer_sizes=[200, 50], corruption_levels=[0.3, 0.2], pretraining_epochs=15, batch_size=5
# 24.45% test error

# hidden_layer_sizes=[200, 100], corruption_levels=[0.3, 0.3], pretraining_epochs=15, batch_size=5
# 25.94% test error

def run_sda_phage(hidden_layer_sizes=[200, 100], corruption_levels=[0.3, 0.3], pretraining_epochs=5, batch_size=5):
    datasets = PrincipledTextFileDataLoader().load_data("dipep_only_others")
    example = datasets[2][0]
    print "type = "
    print type(example)
    ex = example.get_value()[0]
    print ex.shape
    print ex
    print "here's how the example starts..."
    print ex[0]
    print ex[1]
    print ex[2]
    print ex[3]
    #print 1/0
    network = run_SdA(datasets, image_width=20, image_height=20, num_classes=11, 
            hidden_layer_sizes=hidden_layer_sizes, corruption_levels=corruption_levels, 
            pretraining_epochs=pretraining_epochs, batch_size=batch_size)
            
    #print "Datasets" + datasets
    print "Applying the network"
    print network.apply(ex)

if __name__ == '__main__':
    run_sda_phage()
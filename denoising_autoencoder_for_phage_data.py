# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:47:50 2015

@author: jlawson
"""
from principled_text_file_data_loader import PrincipledTextFileDataLoader
from denoising_autoencoder import test_dA

if __name__ == '__main__':
    datasets = PrincipledTextFileDataLoader().load_data("dipep_only_others")
    test_dA(datasets, image_width=20, image_height=20, num_classes=11)


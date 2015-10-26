# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:34:58 2015

@author: jlawson
"""
from deep_belief_network import run_DBN
from principled_text_file_data_loader import PrincipledTextFileDataLoader

def run_dbn_phage():
    datasets = PrincipledTextFileDataLoader(squashed=False).load_data("dipep_only_others")
    run_DBN(datasets, image_width=20, image_height=20, num_classes=11)

if __name__ == '__main__':
    run_dbn_phage()
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:38:24 2019

@author: Daniel Lin

"""

import yaml
import argparse
from keras import backend as K 

# Arguments
parser = argparse.ArgumentParser(description='Training a vulnerability detection model.')
parser.add_argument('--config', type=str, help='Path to the configuration file.')
parser.add_argument('--data_dir', default='data/', type=str, help='The path of the input data for obtaining the representations.')
parser.add_argument('--trained_model', type=str, help='The path of the trained model for test.')
parser.add_argument('--layer', type=int, help='The representations obtained from the n-th layer.')
parser.add_argument('--saved_path', type=str, default='result/', help='The obtained representations are saved in a Pickle object.')
parser.add_argument('--verbose', default=1, help='Show all messages.')
paras = parser.parse_args()
config = yaml.safe_load(open(paras.config,'r'))

from src.helper import GetRepresentation as Helper
    
helper = Helper(config, paras)
helper.exec()

K.clear_session()


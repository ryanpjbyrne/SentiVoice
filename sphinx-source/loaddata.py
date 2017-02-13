
import os
import numpy as np
import random
from io import open

def amazon_reviews():
    datafolder = '/home/ryan/Desktop/sphinx-source/amazon/'
    files = os.listdir(datafolder)
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    for file in files:
        print file
        f = open(datafolder + file, 'r', encoding="utf8")
        label = file
        lines = f.readlines()
        no_lines = len(lines)
        no_training_examples = int(0.7*no_lines)
        print no_lines
        print no_training_examples
        
        f.close()
    return X_train, Y_train, X_test, Y_test
amazon_reviews()


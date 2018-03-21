#
import numpy as np  
import argparse


class ParametersClass(object):

    def __init__(self):
        
        #parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--infile', type=str, default=None, help="")

        self.args = parser.parse_args()
    
    def initialize(self):
        return self.args    

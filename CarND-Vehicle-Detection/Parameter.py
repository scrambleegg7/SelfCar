#
import numpy as np  
import argparse


class ParametersClass(object):

    def __init__(self):
        
        #parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--color', type=str, default="RGB", help="color scheme default RGB")
        parser.add_argument('--hog_channel', type=int, default=0, help="hog_channel default=0, 3=use all channel")

        

        self.args = parser.parse_args()
    
    def initialize(self):
        return self.args    

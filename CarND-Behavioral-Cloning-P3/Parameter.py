#
import numpy as np  
import argparse


class ParametersClass(object):

    def __init__(self):
        
        #parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--straight_angle', default=None, type=float, help="Remove all training data with steering angle less than this. Useful for getting rid of straight bias")
        #parser.add_argument('--save_generated_images', action='store_true', help="Location to save generated images to")
        #parser.add_argument('--load_model', type=str, help="For transfer learning, here's the model to start with")
        parser.add_argument('--directory', type=str, default=None, help="Directory for training data")
        parser.add_argument('--learning_rate', type=float, default=.001)
        #parser.add_argument('--header', type=bool, default=True)
        parser.add_argument('--header', dest='header', action='store_true')
        parser.add_argument('--no-header', dest='header', action='store_false')
        parser.set_defaults(header=True)

        self.args = parser.parse_args()
    
    def initialize(self):
        return self.args    

    def checkParams(self):

        params = self.args
        if params.straight_angle is None:
            print("-"* 30)
            print("ERROR from straight line angle parameter")
            print("-"* 30)
            raise Exception("straight angle is BLANK !!!")

        if params.directory is None:
            print("-"* 30)
            print("ERROR from directory")
            print("-"* 30)
            raise Exception("directory is BLANK !!!")

        
    
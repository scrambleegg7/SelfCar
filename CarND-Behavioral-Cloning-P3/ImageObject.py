

import numpy as np  
import pandas as pd 
import cv2
import os

from skimage.io import imread

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

class ImageDataObjectClass():

    def __init__(self,X,y, cwd, remove_straight_angle=None):

        self.X = X
        self.y = y
        self.cwd = cwd

        self.remove_straight_angle = remove_straight_angle

    def batch_next(self,offset,BATCH_SIZE=32):

        image_samples = self.X[ offset:offset+BATCH_SIZE ]        
        angle_samples = self.y[ offset:offset+BATCH_SIZE ]

        images = []
        angles = []        
        is_flip = False
        for idx, ( image_sample, angle_sample ) in enumerate( zip( image_samples, angle_samples )  ):
            
            actual_filename = image_sample.split("/")[-1]
            name = os.path.join( self.cwd, actual_filename )
            
            if "center" in actual_filename: #  or "right" in actual_filename:
                is_flip = True
            else:
                is_flip = False

            drive_image = self.readImage(name)
            if np.random.random() < 0.5 and is_flip:
                #drive_image = cv2.flip(drive_image, 1)
                drive_image = np.fliplr(drive_image)

                angle_sample *= -1.

            images.append(drive_image)
            angles.append(angle_sample)
        
            #
            # flip image with 50% chance
            #
                

        return shuffle( np.array( images ), np.array( angles ) )

    def readImage(self, file_name):
        #image = cv2.imread(file_name)
        image = imread(file_name)
        return image

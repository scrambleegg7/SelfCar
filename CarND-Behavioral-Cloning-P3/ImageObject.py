

import numpy as np  
import pandas as pd 
import cv2
import os

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

class ImageDataObjectClass():

    def __init__(self,X,y, cwd, remove_straight_angle=None):

        self.X = X
        self.y = y
        self.cwd = cwd

        self.remove_straight_angle = remove_straight_angle

    def crop_img(self,image):
        return image[50:-20,:]

    def getYCrCb(self,image):

        YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return YCrCb

    def resize_img(self,image):

        img = resize(image, (66, 200), mode='reflect')
        return img 

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
            # read process and data augmentation 
            # size changed to 66 x 200 x 3
            #drive_image = self.readImageAndPreProcess(name)
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

    def readImageAndPreProcess(self, file_name):
        #image = cv2.imread(file_name)
        image = imread(file_name)
        # add YCrCb to strength Y.
        image = self.getYCrCb(image)
        image = self.crop_img(image)
        image = self.resize_img(image)

        return image


    def readImage(self, file_name):
        #image = cv2.imread(file_name)
        image = imread(file_name)
        return image

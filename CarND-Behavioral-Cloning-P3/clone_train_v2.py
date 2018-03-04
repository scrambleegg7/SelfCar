#
# Training program designed by H. Hamano version 2
# Mar 2 2018
#

import tensorflow as tf   
import numpy as np  
import pandas as pd 
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.io import imread

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD, Adam, Nadam

from utils import displayImage
from nVidiaModel import nVidiaModelClass
from Parameter import ParametersClass

from keras import backend as K


import argparse

def generator(X, y, baseDir, batch_size=32, remove_straight_angle=None):

    num_samples = X.shape[0]
    cwd = os.path.join(os.getcwd(),baseDir)
    cwd = os.path.join(cwd,"IMG")


    while 1: # Loop forever so the generator never terminates
        

        #
        #    shuffle(samples)
        #
        X, y = shuffle(X, y)

        imageDataObject = ImageDataObject(X,y,cwd, remove_straight_angle)

        for offset in range(0, num_samples, batch_size):

            X_train, y_train = imageDataObject.batch_next(offset,batch_size)

            yield X_train, y_train

class ImageDataObject():

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
            
            if "left" in actual_filename or "center" in actual_filename:
                is_flip = False
            else:
                is_flip = True

            drive_image = self.readImage(name)
            if np.random.random() < 0.5:
                drive_image = cv2.flip(drive_image, 1)
                angle_sample *= -1.

            images.append(drive_image)
            angles.append(angle_sample)
        
            #
            # flip image with 50% chance
            #
                

        return shuffle( np.array( images ), np.array( angles ) )

    def batch_next_old(self,offset,BATCH_SIZE=32):

        side_index = np.random.choice(range(3))
        image_samples = self.X[ offset:offset+BATCH_SIZE,side_index ]
        angle_samples = self.y[ offset:offset+BATCH_SIZE,side_index ]

        images = []
        angles = []        
        for idx, ( image_sample, angle_sample ) in enumerate( zip( image_samples, angle_samples )  ):
            name = os.path.join( self.cwd, image_sample.split("/")[-1] )

            #
            # flip image with 50% chance
            #
            drive_image = self.readImage(name)
            if np.random.random() < 0.5:
                drive_image = cv2.flip(drive_image, 1)
                angle_sample *= -1.


            images.append(drive_image)
            angles.append(angle_sample)
        

        return shuffle( np.array( images ), np.array( angles ) )


    def readImage(self, file_name):
        image = cv2.imread(file_name)
        return image

    

class DataObject():

    def __init__(self):
        pass

    def shuffleSplit(self,test_size=0.3):

        X = self.images
        y = self.steerings
        
        X, y = shuffle(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

        return X_train, X_test, y_train, y_test
        
    def loadCSVData(self, baseDir="data", filename="driving_log.csv", sample = True, remove_straight_angle=None):

        # load 3 angle camera view from flat file
        
        # center
        # left 
        # right 
        #
        # setup directory
        #
        #
        if sample:
            cwd = os.path.join( os.getcwd(), baseDir )
            driving_log = os.path.join(cwd, filename)
            df_drive = pd.read_csv(driving_log)
        else:
            driving_log = os.path.join(baseDir,filename)
            df_drive = pd.read_csv(driving_log,header=None,names=["center","left","right","steering","throttle","brake","speed"])

        # read csv file to save drive data
        centerImages = np.array( df_drive["center"].tolist() ) #[:,np.newaxis]
        leftImages = np.array( df_drive["left"].tolist() )     #[:,np.newaxis]
        rightImages = np.array( df_drive["right"].tolist() )   #[:,np.newaxis]

        #
        #  Mar. 03 2018 changed 0.275 to 0.25
        #               min max function used for left and right angle)
        offset = 0.25
        #   left_angle = min(1.0, center_angle + 0.25)

        Steering_center = np.array( df_drive["steering"].tolist() )    
        Steering_left  = np.array( list( map(lambda steer:(steer + offset), Steering_center.copy() ) )  )  
        Steering_right = np.array( list( map(lambda steer:(steer - offset), Steering_center.copy() ) ) )  

        if remove_straight_angle is not None:
            print("Exclude straight line images and angles from training / validation data..")
            print("Omit Angle less than %.2f" % remove_straight_angle )

        #
        # remove straight angle
        #
        # make mask
        #
        mask_center = Steering_center > remove_straight_angle
        mask_left = Steering_left > remove_straight_angle
        mask_right = Steering_right > remove_straight_angle

        # mask for small angle
        mask_center_small_angle = (mask_center == False)
        mask_left_small_angle = (mask_left == False)
        mask_right_small_angle = (mask_right == False)        

        #
        shrink_rate = 0.3
        #
        mask_center_20 = int( np.sum(mask_center_small_angle) * shrink_rate )
        mask_left_20 = int( np.sum(mask_left_small_angle) * shrink_rate )
        mask_right_20 = int( np.sum(mask_right_small_angle) * shrink_rate )

        # drop off  straight line data
        Steering_center_small = np.random.choice(Steering_center[mask_center_small_angle], mask_center_20 )  
        Steering_left_small = np.random.choice(Steering_left[mask_left_small_angle], mask_left_20 ) 
        Steering_right_small = np.random.choice(Steering_right[mask_right_small_angle], mask_right_20 ) 

        # concatenate steep steering angle + some straight line data
        Steering_center = np.concatenate( [ Steering_center[mask_center], Steering_center_small ] )
        Steering_left = np.concatenate( [ Steering_left[mask_left], Steering_left_small ] )
        Steering_right = np.concatenate( [Steering_right[mask_right], Steering_right_small ] )

        #
        # Images file name 
        #
        centerImages_small = centerImages[mask_center_small_angle]
        leftImages_small = leftImages[mask_left_small_angle]
        rightImages_small = rightImages[mask_right_small_angle]

        centerImages = centerImages[mask_center]
        leftImages = leftImages[mask_left]
        rightImages = rightImages[mask_right]

        centerImages_small = np.random.choice( centerImages_small , mask_center_20 )
        leftImages_small = np.random.choice( leftImages_small , mask_left_20 )   
        rightImages_small = np.random.choice( rightImages_small , mask_right_20 )   

        centerImages = np.concatenate([ centerImages, centerImages_small ])
        leftImages = np.concatenate([ leftImages, leftImages_small ])
        rightImages = np.concatenate([ rightImages, rightImages_small ])

        print("center Image",centerImages.shape)
        print("left Image", leftImages.shape)
        print("right Image", rightImages.shape)
        
        print("center steering",Steering_center.shape)
        print("left steering",Steering_left.shape)
        print("right steering",Steering_right.shape)

        self.images = np.concatenate( [centerImages,leftImages,rightImages ], axis=0  )
        self.steerings = np.concatenate([Steering_center,Steering_left, Steering_right ],axis=0)

        #return images, steerings
def parameters():

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--remove_straight_angle', default=None, type=float, help="Remove all training data with steering angle less than this. Useful for getting rid of straight bias")
    parser.add_argument('--save_generated_images', action='store_true', help="Location to save generated images to")
    parser.add_argument('--load_model', type=str, help="For transfer learning, here's the model to start with")
    parser.add_argument('--directory', type=str, default=None, help="Directory for training data")
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--header', dest='header', action='store_true')
    parser.add_argument('--no-header', dest='header', action='store_false')
    parser.set_defaults(header=True)

    args = parser.parse_args()
    return args    

def main():

    BATCH_SIZE = 64
    #(after you are done with the model)
    #K.clear_session()
    paramCls = ParametersClass()
    params = paramCls.initialize()
    paramCls.checkParams()

    filename = "driving_log.csv"
    baseDir = params.directory
    learning_rate = params.learning_rate

    print("-"*30)
    print("* EPOCHS *",params.epochs)
    print("* Learning RATE *",params.learning_rate)
    print("* load sample data", params.header)
    print("-"*30)
 
    myData = DataObject()
    myData.loadCSVData(baseDir, filename, sample = params.header, remove_straight_angle=params.remove_straight_angle)
    X_train, X_test, y_train, y_test = myData.shuffleSplit(test_size=0.3)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    nVidia = nVidiaModelClass()
    model = nVidia.buildModel_drop()

    train_generator = generator(X_train, y_train, baseDir, batch_size=BATCH_SIZE)    #, remove_straight_angle=params.remove_straight_angle)
    validation_generator = generator(X_test, y_test, baseDir, batch_size=BATCH_SIZE) #, remove_straight_angle=params.remove_straight_angle)

    optimizer = Nadam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    if keras.__version__ == "1.2.1":
        model.fit_generator(train_generator, samples_per_epoch=X_train.shape[0], \
                validation_data=validation_generator,  \
                nb_val_samples=X_test.shape[0], nb_epoch=params.epochs, verbose=1)
    else:
        steps_per_epch = X_train.shape[0] // BATCH_SIZE
        valid_steps = X_test.shape[0] // BATCH_SIZE 
        model.fit_generator(train_generator, steps_per_epoch=steps_per_epch , \
            validation_data=validation_generator,  \
            validation_steps= valid_steps  , epochs=params.epochs, verbose=1)
        
    #X,y = (next(train_generator))
    #print(X.shape, y.shape)
    #displayImage(X,y)
    model_file = os.path.join("convmodel","model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()
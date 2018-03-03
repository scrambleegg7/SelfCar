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

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD, Adam, Nadam

from utils import displayImage
from nVidiaModel import nVidiaModelClass

from keras import backend as K


import argparse

def generator(X, y, baseDir, batch_size=32):

    num_samples = X.shape[0]
    cwd = os.path.join(os.getcwd(),baseDir)
    cwd = os.path.join(cwd,"IMG")
    
    while 1: # Loop forever so the generator never terminates
        

        #
        #    shuffle(samples)
        #
        X, y = shuffle(X, y)

        imageDataObject = ImageDataObject(X,y,cwd)

        for offset in range(0, num_samples, batch_size):

            X_train, y_train = imageDataObject.batch_next(offset,batch_size)

            yield X_train, y_train

class ImageDataObject():

    def __init__(self,X,y, cwd):

        self.X = X
        self.y = y
        self.cwd = cwd

    def batch_next(self,offset,BATCH_SIZE=32):

        side_index = np.random.choice(range(3))
        image_samples = self.X[ offset:offset+BATCH_SIZE,side_index ]
        angle_samples = self.y[ offset:offset+BATCH_SIZE,side_index ]

        images = []
        angles = []        
        for idx, ( image_sample, angle_sample ) in enumerate( zip( image_samples, angle_samples )  ):
            name = os.path.join( self.cwd, image_sample.split("/")[-1] )
            
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
        
    def loadCSVData(self, baseDir="data", filename="driving_log.csv", sample = True):

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
        centerImages = np.array( df_drive["center"].tolist() )[:,np.newaxis]
        leftImages = np.array( df_drive["left"].tolist() )[:,np.newaxis]
        rightImages = np.array( df_drive["right"].tolist() )[:,np.newaxis]

        self.images = np.concatenate( [centerImages,leftImages,rightImages ], axis=1  )

        offset = 0.275
        Steering_center = np.array( df_drive["steering"].tolist() )    
        Steering_left  = np.array( list( map(lambda steering:steering + offset  ,Steering_center.copy() ) )  )  
        Steering_right = np.array(  list( map(lambda steering:steering - offset  ,Steering_center.copy() ) ) )  

        self.steerings = np.concatenate([Steering_center[:,np.newaxis],Steering_left[:,np.newaxis], Steering_right[:,np.newaxis] ],axis=1)

        #return images, steerings
def parameters():

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--remove_straight_angle', default=None, type=float, help="Remove all training data with steering angle less than this. Useful for getting rid of straight bias")
    parser.add_argument('--save_generated_images', action='store_true', help="Location to save generated images to")
    parser.add_argument('--load_model', type=str, help="For transfer learning, here's the model to start with")
    parser.add_argument('--directory', type=str, default='data', help="Directory for training data")
    parser.add_argument('--learning_rate', type=float, default=.001)
    args = parser.parse_args()
    return args    

def main():

    #(after you are done with the model)
    K.clear_session()

    filename = "driving_log.csv"
    baseDir = "data"
    learning_rate = 0.001

    myData = DataObject()
    myData.loadCSVData(baseDir, filename, sample = True)
    X_train, X_test, y_train, y_test = myData.shuffleSplit(test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    nVidia = nVidiaModelClass()
    model = nVidia.buildModel()

    train_generator = generator(X_train, y_train, baseDir, batch_size=128)
    validation_generator = generator(X_test, y_test, baseDir, batch_size=128)

    optimizer = Nadam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    samples_p_epoch = (np.ceil(X_train.shape[0] / 128) + 1) * 128

    model.fit_generator(train_generator, samples_per_epoch=samples_p_epoch, \
             validation_data=validation_generator,  \
            nb_val_samples=X_test.shape[0], nb_epoch=3, verbose=1)
    #X,y = (next(train_generator))
    #print(X.shape, y.shape)
    #displayImage(X,y)
    model_file = os.path.join("convmodel","model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()
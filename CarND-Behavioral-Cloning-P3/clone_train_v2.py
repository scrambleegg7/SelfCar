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

        for offset in range(0, num_samples, batch_size):

            # decide which side should be trained. center 0, left 1, right 2
            ridx = np.random.choice(range(3))

            image_samples = X[offset:offset+batch_size,ridx]
            angle_samples = y[offset:offset+batch_size, ridx]

            #batch_samples = samples[offset:offset+batch_size]

            images = []
            #angles = []
            for idx, ( image_sample, angle_sample ) in enumerate( zip( image_samples, angle_samples )  ):
                name = os.path.join( cwd, image_sample.split("/")[-1] )
                #print("fname:",name)
                drive_image = cv2.imread(name)
                images.append(drive_image)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = angle_samples
            yield X_train, y_train


class ImageDataObject():

    def readImage(self):

        pass

class ModelObject():

    def createPreProcessingLayers(self):
        """
        Creates a model with the initial pre-processing layers.
        """
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

        # cropping image size 50px from top ~ 20 px from bottom 
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        return model
    
    def nVidiaModel(self):
        """
        Creates nVidea Autonomous Car Group model
        """
        model = self.createPreProcessingLayers()
        #
        # suppress kera v.2 warning message Conv2d should be used.
        #
        #model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        #model.add(Convolution2D(64,3,3, activation='relu'))
        #model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu',name="conv1"))
        model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu',name="conv2"))
        model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu',name="conv3"))
        model.add(Conv2D(64,(3,3), activation='relu',name="conv4"))
        model.add(Conv2D(64,(3,3), activation='relu',name="conv5"))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model   

class DataObject():

    def __init__(self):
        pass

    def shuffleSplit(self,X=None,y=None):

        if X == None and y == None:
            X = self.images
            y = self.steerings
        
        X, y = shuffle(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

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
    X_train, X_test, y_train, y_test = myData.shuffleSplit()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = ModelObject().nVidiaModel()
    #model.summary()

    train_generator = generator(X_train, y_train, baseDir, batch_size=64)
    validation_generator = generator(X_test, y_test, baseDir, batch_size=64)

    optimizer = Nadam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit_generator(train_generator, samples_per_epoch= \
            X_train.shape[0], validation_data=validation_generator, \
            nb_val_samples=X_test.shape[0], nb_epoch=3)
    #X,y = (next(train_generator))
    #print(X.shape, y.shape)
    #displayImage(X,y)
    model_file = os.path.join("convmodel","model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()
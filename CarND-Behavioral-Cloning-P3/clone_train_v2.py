#
# Training program designed by H. Hamano version 2
# Mar 2 2018
#

import tensorflow as tf   
import numpy as np  
import pandas as pd 
import cv2
import os
from skimage.io import imread

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def generator(X, y, batch_size=32):

    num_samples = X.shape[0]
    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        
        X, y = shuffle(X, y)

        for offset in range(0, num_samples, batch_size):

            # decide which side should be trained.
            ridx = np.random.choice(range(3))

            image_samples = X[offset:offset+batch_size,ridx]
            angle_samples = y[offset:offset+batch_size, ridx]

            #batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for idx, image_sample in image_samples:
                name = './IMG/'+image_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                #center_angle = float(batch_sample[3])
                images.append(center_image)
                #angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = angle_samples
            yield shuffle(X_train, y_train)




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
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        return model
    
    def nVidiaModel(self):
        """
        Creates nVidea Autonomous Car Group model
        """
        model = self.createPreProcessingLayers()
        model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(Convolution2D(64,3,3, activation='relu'))
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
        
    def loadCSVData(self, baseDir="./data", filename="driving_log.csv", sample = False):

        # load 3 angle camera view from flat file
        
        # center
        # left 
        # right 
        #
        # setup directory
        #
        #
        if sample:
            driving_log = os.path.join(baseDir,filename)
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

def main():

    filename = "driving_log.csv"
    baseDir = "./data"

    myData = DataObject()
    myData.loadCSVData(baseDir, filename, sample = True)
    X_train, X_test, y_train, y_test = myData.shuffleSplit()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = ModelObject().nVidiaModel()
    model.summary()

    trainGenerator = generator(X_train, y_train)

if __name__ == "__main__":
    main()
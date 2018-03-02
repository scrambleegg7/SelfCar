#
# Training program designed by H. Hamano
# Feb 26 2018
#

import tensorflow as tf   
import numpy as np  
import pandas as pd 
import cv2

import os
from skimage.io import imread

from keras.models import Sequential
from keras.layers import Flatten, Dense   


def nVidiaModel(model):
    """
    Creates nVidea Autonomous Car Group model
    """
    #model = createPreProcessingLayers()
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

def loadData(sample=False):

    baseDir = "./data"
    driving_log_file = "driving_log.csv"
    driving_log = os.path.join(baseDir,driving_log_file)

    if sample:
        df_drive = pd.read_csv(driving_log)
    else:
        df_drive = pd.read_csv(driving_log,header=None,names=["center","left","right","steering","throttle","brake","speed"])

    # read csv file to save drive data
    centerImagesPath = df_drive["center"].tolist()
    Steering = df_drive["steering"].tolist()    

    assert( len(centerImagesPath) == len(Steering)) 

    if sample:
        images = list(map(lambda file:imread( os.path.join(baseDir, file)   ) , centerImagesPath  ))
    else:
        images = list(map(lambda file:imread( file   ) , centerImagesPath  ))
        
    return np.array(images), np.array(Steering)

def train():

    X_train, y_train = loadData(sample=True)

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)

    imshape = X_train.shape

    model = Sequential()
    model.add( Flatten( input_shape=imshape[1:] )  )
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train,y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

    model.save("model.h5")

def main():


    train()



if __name__ == "__main__":

    main()
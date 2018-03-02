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

from sklearn.utils import shuffle

from utils import displayImage
from nVidiaModel import nVidiaModelClass

def loadData(sample=False):

    baseDir = "data"
    filename = "driving_log.csv"
    
    cwd = os.path.join( os.getcwd(), baseDir )
    driving_log = os.path.join(cwd, filename)

    if sample:
        df_drive = pd.read_csv(driving_log)
    else:
        df_drive = pd.read_csv(driving_log,header=None,names=["center","left","right","steering","throttle","brake","speed"])

    # read csv file to save drive data
    centerImagesPath = df_drive["center"].tolist()
    Steering = df_drive["steering"].tolist()    

    assert( len(centerImagesPath) == len(Steering)) 

    if sample:
        print("Read sample data..")
        imageDir = os.path.join(cwd,"IMG")
        images = list(map(lambda file:imread( os.path.join(imageDir, file.split("/")[-1]   )   ) , centerImagesPath  ))
    else:
        images = list(map(lambda file:imread( file   ) , centerImagesPath  ))
        
    return np.array(images), np.array(Steering)

def train():

    X_train, y_train = loadData(sample=True)

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)

    X_train, y_train = shuffle(X_train,y_train)

    #displayImage( X_train[:32], y_train[:32]  )


    model = nVidiaModelClass().nVidiaModel()
    #model = Sequential()
    #model.add( Flatten( input_shape=(160,320,3) ) ) 
    #model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train,y_train, validation_split=0.3, shuffle=True, nb_epoch=2)

    model_file = os.path.join("testmodel","model.h5")
    model.save(model_file)

def main():


    train()



if __name__ == "__main__":

    main()
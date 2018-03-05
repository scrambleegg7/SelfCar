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
from keras import backend as K

from utils import displayImage
from nVidiaModel import nVidiaModelClass
from Parameter import ParametersClass
from DataObject import DataObjectClass
from ImageObject import ImageDataObjectClass

def generator(X, y, baseDir, batch_size=32, straight_angle=None):

    num_samples = X.shape[0]
    cwd = os.path.join(os.getcwd(),baseDir)
    cwd = os.path.join(cwd,"IMG")


    while 1: # Loop forever so the generator never terminates
        

        #
        #    shuffle(samples)
        #
        X, y = shuffle(X, y)

        imageDataObject = ImageDataObjectClass(X,y,cwd, straight_angle)

        for offset in range(0, num_samples, batch_size):

            X_train, y_train = imageDataObject.batch_next(offset,batch_size)

            yield X_train, y_train

def main():

    BATCH_SIZE = 128
    #(after you are done with the model)
    #K.clear_session()
    paramCls = ParametersClass()
    params = paramCls.initialize()
    paramCls.checkParams()

    filename = "driving_log.csv"
    baseDir = params.directory
    
    print("-"*30)
    print("* EPOCHS *",params.epochs)
    print("* Learning RATE *",params.learning_rate)
    print("* load sample data", params.header)
    print("-"*30)
 
    myData = DataObjectClass()
    myData.loadCSVData(baseDir, filename, sample = params.header, straight_angle=params.straight_angle)
    X_train, X_test, y_train, y_test = myData.shuffleSplit(test_size=0.15)
    
    print(" Train / Test splitted size --> ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    nVidia = nVidiaModelClass()
    model = nVidia.buildModel()

    train_generator = generator(X_train, y_train, baseDir, batch_size=BATCH_SIZE)    #, straight_angle=params.straight_angle)
    validation_generator = generator(X_test, y_test, baseDir, batch_size=BATCH_SIZE) #, straight_angle=params.straight_angle)

    #optimizer = Nadam(lr=learning_rate)
    optimizer = Adam(lr=params.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    if keras.__version__ == "1.2.1":
        history_object = model.fit_generator(train_generator, samples_per_epoch=X_train.shape[0], \
                validation_data=validation_generator,  \
                nb_val_samples=X_test.shape[0], nb_epoch=params.epochs, verbose=1)
    else:
        steps_per_epch = X_train.shape[0] // BATCH_SIZE
        valid_steps = X_test.shape[0] // BATCH_SIZE 
        history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epch , \
            validation_data=validation_generator,  \
            validation_steps= valid_steps  , epochs=params.epochs, verbose=1)


    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model_file = os.path.join("convmodel","model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()
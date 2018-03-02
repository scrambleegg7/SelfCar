
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD, Adam, Nadam


class nVidiaModelClass():

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
#
# nVidiaModel
#
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Conv2D
from keras.layers import Dropout, Activation
from keras.regularizers import l2 # activity_l2
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD, Adam, Nadam


class nVidiaModelClass():

    def __init__(self):

        print(keras.__version__)
        self.kversion = keras.__version__

        #self.buildModel()

    def createPreProcessingLayers(self):
        """
        Creates a model with the initial pre-processing layers.
        """
        model = Sequential()
        model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160,320,3)))

        # cropping image size 50px from top ~ 20 px from bottom 
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        return model

    def createNormalizedLayers(self):
        """
        Creates a model with the initial pre-processing layers.
        """
        # image is shrinked size image 66 x 200 x 3 YCrCb image
        model = Sequential()
        model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(66,200,3)))

        # cropping image size 50px from top ~ 20 px from bottom 
        #model.add(Cropping2D(cropping=((50,20), (0,0))))
        return model

    def buildModel(self):
        """
        Creates nVidea Autonomous Car Group model
        """
        model = self.createPreProcessingLayers()

        if self.kversion == "1.2.1":        
            #
            # suppress kera v.2 warning message Conv2d should be used.
            #
            model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
            model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
            model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
            model.add(Convolution2D(64,3,3, activation='relu'))
            model.add(Convolution2D(64,3,3, activation='relu'))
        else:
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

    def buildModel_Normal(self):
        """
        Creates nVidea Autonomous Car Group model
        """
        model = self.createPreProcessingLayers()
        if self.kversion == "1.2.1":        
            #
            # suppress kera v.2 warning message Conv2d should be used.
            #
            model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
            model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
            model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
            model.add(Convolution2D(64,3,3, activation='elu'))
            model.add(Convolution2D(64,3,3, activation='elu'))
        else:
            model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu',name="conv1"))
            model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu',name="conv2"))
            model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu',name="conv3"))
            model.add(Conv2D(64,(3,3), activation='elu',name="conv4"))
            model.add(Conv2D(64,(3,3), activation='elu',name="conv5"))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model   


    def buildModel_drop(self):
        """
        Creates nVidea Autonomous Car Group model
        """
        model = self.createPreProcessingLayers()

        if self.kversion == "1.2.1":        
            #
            # suppress kera v.2 warning message Conv2d should be used.
            #

            #  31 x 98 x 24
            model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu', init="glorot_normal", W_regularizer=l2(0.001)) )
            model.add(Dropout(0.1))  # keep_prob 0.9
            # 14 x 47 x 36
            model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu', init="glorot_normal", W_regularizer=l2(0.001)))
            model.add(Dropout(0.2))  # keep_prob 0.8
            # 5 x 22 x 48
            model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu', init="glorot_normal", W_regularizer=l2(0.001)))
            model.add(Dropout(0.2))  # keep_prob 0.8
            # 3 x 20 x 64
            model.add(Convolution2D(64,3,3, subsample=(1,1),activation='elu', init="glorot_normal", W_regularizer=l2(0.001)))
            model.add(Dropout(0.2))  # keep_prob 0.8
            # 1 x 18 x 64
            model.add(Convolution2D(64,3,3, subsample=(1,1),activation='elu', init="glorot_normal", W_regularizer=l2(0.001)))
            #model.add(Dropout(0.2))  # keep_prob 0.8
            model.add(Flatten())
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(100,activation='elu', init='glorot_normal', W_regularizer=l2(0.001)))
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(50,activation='elu', init='glorot_normal', W_regularizer=l2(0.001)))
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(10,activation='elu', init='glorot_normal', W_regularizer=l2(0.001)))
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(1,activation='linear', init='glorot_normal'))

        else:
            model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu',kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), name="conv1"))
            model.add(Dropout(0.1)) # keep_rate 0.9
            model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu',kernel_initializer="he_uniform", kernel_regularizer=l2(0.01),  name="conv2"))
            model.add(Dropout(0.2)) # keep_rate 0.8
            model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu',kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), name="conv3"))
            model.add(Dropout(0.2)) # keep_rate 0.8            
            model.add(Conv2D(64,(3,3), activation='elu',kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), name="conv4"))
            model.add(Dropout(0.2)) # keep_rate 0.8
            model.add(Conv2D(64,(3,3), activation='elu',kernel_initializer="he_uniform", kernel_regularizer=l2(0.01), name="conv5"))
        
            model.add(Flatten())
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(100,activation='elu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01) ))
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(50,activation='elu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01) ) )
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(10,activation='elu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01) ) )
            model.add(Dropout(0.5))  # keep_prob 0.5
            model.add(Dense(1,activation='linear', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01) ))

        return model   


def main():
    nVidia = nVidiaModelClass()
    model = nVidia.buildModel()
    model.summary()
if __name__ == "__main__":
    main()    
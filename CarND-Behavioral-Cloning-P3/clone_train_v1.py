

def get_model():
    """
    Returns a compiled keras model ready for training.
    """
    ##
    ## nVIDIA model
    ##
    model = Sequential([
        # Normalize image to -1.0 to 1.0
        Lambda(normalize, input_shape=(66, 200, 3)),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation 
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .1 (keep probability of .9)
        Dropout(.1),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Flatten
        Flatten(),
        # Dropout with drop probability of .3 (keep probability of .7)
        Dropout(.3),
        # Fully-connected layer 1 | 100 neurons | elu activation
        Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 2 | 50 neurons | elu activation
        Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 3 | 10 neurons | elu activation
        Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Output
        Dense(1, activation='linear', init='he_normal')
    ])

    model.compile(optimizer='adam', loss='mse')
return model 
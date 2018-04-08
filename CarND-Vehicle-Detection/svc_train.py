
#
from utils_vehicles import *
from trainDataClass import TrainDataClass

from Parameter import ParametersClass

import numpy as np

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import pickle

from Parameter import ParametersClass

paramCls = ParametersClass()
params = paramCls.initialize()


def build_svc_model(color, hog_channel_):



    trainData = TrainDataClass()

    car_images = trainData.car_images
    non_car_images = trainData.non_car_images

    car_features = build_features(car_images, color_space=color, spatial_size=(32, 32),
                                    hist_bins=32, orient=9, 
                                    pix_per_cell=8, cell_per_block=2, hog_channel=hog_channel_,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)

    non_car_features = build_features(non_car_images, color_space=color, spatial_size=(32, 32),
                                    hist_bins=32, orient=9, 
                                    pix_per_cell=8, cell_per_block=2, hog_channel=hog_channel_,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)


    print(" loaded car features -->",  len(car_features), car_features[0].shape)
    print(" loaded non car features -->",  len(non_car_features) , non_car_features[0].shape  )


    # first of all build label data for y

    # 1 ---> CAR
    # 
    # 0 ---> NON CAR 
    #
    # total length 8792 + 8968 = 17760
    #

    # Define a labels vector based on features lists
    print("  Buildng X y training data set .....")
    y = np.hstack((np.ones(len(car_features)), 
                np.zeros(len(non_car_features))))
    # 
    # next, build X training data 
    #
    X = np.vstack((car_features, non_car_features)).astype(np.float64)    

    print("X shape", X.shape)                    
    ## Train / Test Data split 
    # Split up data into randomized training and test sets
    print(" Spliting data set to train == 80% / test == 20%")
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print(" save X / y train test data ....")
    np.save("X_train.npy", X_train  )
    np.save("X_test.npy", X_test  )
    np.save("y_train.npy", y_train  )
    np.save("y_test.npy", y_test  )
    
    

    print(" Scaling (Normalized data) ....")
    # standarized (normalized data)
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to both X_train and X_test
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    print(" Suuport Vector training starting ....")
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    print(" Building Model ....")
    
    svc.fit(scaled_X_train, y_train)

    print(" Ended model making ....")
    print(" Check scoring ....")
    print('Test Accuracy of SVC from building model  = ', svc.score(scaled_X_test, y_test))

    return svc

def main():

    print("-"*30)
    print("* Color Scheme ... *",params.color)
    print("* HOG channel ... *",params.hog_channel)
    print("-"*30)    

    color = params.color
    hog_channel_ = params.hog_channel
    

    model = build_svc_model(color,hog_channel_)

    # save the classifier
    with open('my_svc_classifier.pkl', 'wb') as fid:
        pickle.dump(model, fid)    


if __name__ == "__main__":
    main()



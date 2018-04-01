#
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
from skimage.io import imread

from Parameter import ParametersClass

from svc_train import build_svc_model
from sklearn.preprocessing import StandardScaler

from utils_vehicles import *

paramCls = ParametersClass()
params = paramCls.initialize()


def load_model():

    # and later you can load it
    print("loading svc model ....")

    with open('my_svc_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    return clf

def load_X_train():

    X_train = np.load("X_train.npy")

    return X_train
    
def main_search():

    # build model based on car / non car image data set
    # model is build with Support Vector machien learning.
    #
    clf = load_model()

    X_train = load_X_train()
    print(" Scaling (Normalized data) ....")
    # standarized (normalized data)
    X_scaler = StandardScaler().fit(X_train)

    test_base = "./test_images"
    
    test_files = []
    target_dir = os.path.join( test_base , "*")
    files = glob.glob( target_dir  )
    test_files.extend(files)

    print("Number of Test Data ..", len(test_files))

    imread_ops = lambda im : imread(im)
    test_images  = np.array( list( map( imread_ops , test_files  )))

    print(" test image size .." , test_images[0].shape)

    for image in test_images:
        window_list = slide_window(image)
        print(" window list length ..",  len( window_list ))

        draw_image = np.copy(image)
        #
        # YCrCb / need full 3 channels used for HOG features
        #  
        hot_windows = search_windows(image, window_list, clf, X_scaler, color_space='YCrCb', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=3, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)             
        plt.imshow(window_img)
        plt.show()


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = build_image_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)

        #7) If positive (prediction == 1) then save the window
        # prediction == 1 means CAR 
        #
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    

def main():

    main_search()


if __name__ == "__main__":
    main()
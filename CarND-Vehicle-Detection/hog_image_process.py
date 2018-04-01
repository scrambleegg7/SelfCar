import cv2
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
import time 

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from utils_vehicles import *


class TrainDataClass(object):

    def __init__(self):
    
        self.cars = []
        self.non_cars = []

        self.read_images()

        self.ops_set()
        self.hog_ops_set()

    

    def load_images(self):

        vehicle_base = "./vehicles"
        vehicle_dirs = os.listdir(vehicle_base)
        vehicle_dirs = [v for v in vehicle_dirs if "DS_Store" not in v]


        for im_type in vehicle_dirs:
            target_dir = os.path.join( vehicle_base, im_type )
            target_dir = os.path.join( target_dir , "*")
            files = glob.glob( target_dir  )
            self.cars.extend(files)

        print("Number of Cars Data ..", len(self.cars))

        non_vehicle_base = "./non-vehicles"
        non_vehicle_dirs = os.listdir(non_vehicle_base)
        non_vehicle_dirs = [v for v in non_vehicle_dirs if "DS_Store" not in v]


        for im_type in non_vehicle_dirs:
            target_dir = os.path.join( non_vehicle_base, im_type )
            target_dir = os.path.join( target_dir , "*")
            files = glob.glob( target_dir  )
            self.non_cars.extend(files)

        print("Number of Non Cars Data ..", len(self.non_cars))


        self.car_filename = [car.split("/")[-1] for car in self.cars]
        self.non_car_filename = [ncar.split("/")[-1] for ncar in self.non_cars]


    def read_images(self):

        self.load_images()

        imread_ops = lambda im : imread(im)
        car_images  = np.array( list( map( imread_ops , self.cars  )))
        non_car_images = np.array( list( map(imread_ops, self.non_cars )) )    

        # for testing purposes 
        # just pick up top10 images 

        rc = np.random.permutation( len( self.cars) )
        rnc = np.random.permutation( len(self.non_cars ))

        rc = rc[:10]
        rnc = rnc[:10]

        self.car_images = car_images[rc]
        self.non_car_images = non_car_images[rnc]
    
    def ops_set(self):
        self.gray_ops = lambda im: cv2.cvtColor(im , cv2.COLOR_RGB2GRAY)
        self.YCrCb_ops = lambda im : cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)

    def hog_ops_set(self):
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        self.hog_ops = lambda im:get_hog_features(im, orient, 
                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)



    def hog_ycrcb(self):

        YCrCb_images = list(map(self.YCrCb_ops,self.car_images)   )
        ncYCrCb_images = list(map(self.YCrCb_ops,self.non_car_images)   )

        YCrCb_images_0 = [i[:,:,0] for i in YCrCb_images]
        YCrCb_images_1 = [i[:,:,1] for i in YCrCb_images]
        YCrCb_images_2 = [i[:,:,2] for i in YCrCb_images]

        ncYCrCb_images_0 = [i[:,:,0] for i in ncYCrCb_images]
        ncYCrCb_images_1 = [i[:,:,1] for i in ncYCrCb_images]
        ncYCrCb_images_2 = [i[:,:,2] for i in ncYCrCb_images]

        hog_results_y = np.array( list(map( self.hog_ops, YCrCb_images_0 )) )
        hog_results_cr = np.array( list(map( self.hog_ops, YCrCb_images_1 )) )
        hog_results_cb = np.array( list(map( self.hog_ops, YCrCb_images_2 )) )

        nchog_results_y = np.array( list(map( self.hog_ops, ncYCrCb_images_0 )) )
        nchog_results_cr = np.array( list(map( self.hog_ops, ncYCrCb_images_1 )) )
        nchog_results_cb = np.array( list(map( self.hog_ops, ncYCrCb_images_2 )) )

        # Setting data image label 
        y_n_f = ["Y ch.: " + f for f in self.car_filename]
        y_h_f = ["Y HOG: " + f for f in self.car_filename]
        cr_n_f = ["Cr ch.: " + f for f in self.car_filename]
        cr_h_f = ["Cr HOG: " + f for f in self.car_filename]
        cb_n_f = ["Cb ch.: " + f for f in self.car_filename]
        cb_h_f = ["Cb HOG: " + f for f in self.car_filename]

        ncy_n_f = ["Y ch.: " + f for f in self.non_car_filename]
        ncy_h_f = ["Y HOG: " + f for f in self.non_car_filename]
        nccr_n_f = ["Cr ch.: " + f for f in self.non_car_filename]
        nccr_h_f = ["Cr HOG: " + f for f in self.non_car_filename]
        nccb_n_f = ["Cb ch.: " + f for f in self.non_car_filename]
        nccb_h_f = ["Cb HOG: " + f for f in self.non_car_filename]

        # apply hog  
        hog_results_y = np.array( list(map( self.hog_ops, YCrCb_images_0 )) )
        hog_results_cr = np.array( list(map( self.hog_ops, YCrCb_images_1 )) )
        hog_results_cb = np.array( list(map( self.hog_ops, YCrCb_images_2 )) )

        nchog_results_y = np.array( list(map( self.hog_ops, ncYCrCb_images_0 )) )
        nchog_results_cr = np.array( list(map( self.hog_ops, ncYCrCb_images_1 )) )
        nchog_results_cb = np.array( list(map( self.hog_ops, ncYCrCb_images_2 )) )

        images_list =  np.asarray(list( zip( YCrCb_images_0, hog_results_y[:,1],YCrCb_images_1, hog_results_cr[:,1],YCrCb_images_2, hog_results_cb[:,1] ) )     )   
        images_label =  np.asarray(list( zip( y_n_f, y_h_f, cr_n_f, cr_h_f, cb_n_f, cb_h_f  ) ))

        showImageList(images_list, images_label,cols=6,fig_size=(10, 8) )


def main():

    trainData = TrainDataClass()
    #trainData.hog_ycrcb()

if __name__ == "__main__":
    main()
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

from utils import *


def load_images():

    vehicle_base = "./vehicles"
    vehicle_dirs = os.listdir(vehicle_base)
    vehicle_dirs = [v for v in vehicle_dirs if "DS_Store" not in v]

    cars = []
    for im_type in vehicle_dirs:
        target_dir = os.path.join( vehicle_base, im_type )
        target_dir = os.path.join( target_dir , "*")
        files = glob.glob( target_dir  )
        cars.extend(files)

    print("Number of Cars Data ..", len(cars))

    non_vehicle_base = "./non-vehicles"
    non_vehicle_dirs = os.listdir(non_vehicle_base)
    non_vehicle_dirs = [v for v in non_vehicle_dirs if "DS_Store" not in v]

    non_cars = []
    for im_type in non_vehicle_dirs:
        target_dir = os.path.join( non_vehicle_base, im_type )
        target_dir = os.path.join( target_dir , "*")
        files = glob.glob( target_dir  )
        non_cars.extend(files)

    print("Number of Non Cars Data ..", len(non_cars))

    return cars, non_cars

def read_images():

    cars, non_cars = load_images()

    imread_ops = lambda im : imread(im)
    car_images  = np.array( list( map( imread_ops , cars  )))
    non_car_images = np.array( list( map(imread_ops, non_cars )) )    

    # for testing purposes 
    # just pick up top10 images 

    rc = np.random.permutation( len( cars) )
    rnc = np.random.permutation( len(non_cars ))

    rc = rc[:10]
    rnc = rnc[:10]

    car_images = car_images[rc]
    non_car_images = non_car_images[rnc]




def main():

    read_images()

if __name__ == "__main__":
    main()
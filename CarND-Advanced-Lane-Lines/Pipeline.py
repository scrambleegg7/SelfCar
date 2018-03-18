import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error

import pickle
import sys

import os
import pandas as pd
import matplotlib.gridspec as gridspec

import seaborn as sns

from PIL import Image
from skimage.transform import rescale, resize, rotate
from skimage.color import gray2rgb, rgb2gray
from skimage import transform, filters, exposure
from skimage.io import imread, imsave

from scipy.ndimage.interpolation import rotate
import platform

from utils import showImageList
from utils import pipelineBinaryImage

gray_ops = lambda img:cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

class Pipeline(object):

    def __init__(self):

        # load normal image data
        self.images = self.loadImageData()
        # load camera calibration parameters
        self.loadCalibration()
        # build undistortion images with parameters
        self.make_undistortion(self.images,self.mtx, self.dist)


    def loadCalibration(self):

        dist_pickle = pickle.load( open( "./pickled_data/calibration.p", "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

    def loadImageData(self):
        self.filenames = sorted(os.listdir("./test_images/") )
        return list( map( lambda x: imread( os.path.join("./test_images",x)), self.filenames) )

    def make_undistortion(self, images, mtx, dist):
        self.undist_images = list(map(lambda image:cv2.undistort(image, mtx, dist, None, mtx), np.copy(images) ) )   


        self.undist_images_gray = list(map(gray_ops, np.copy(self.undist_images) ) )   

    def showImageOnDisplay(self,images_list,images_label,cols=2,fig_size=(8, 14)):

        showImageList(images_list, images_label,cols=cols,fig_size=fig_size )

    def displayNormalUndistort(self):

        undistort_filenames = list(map( lambda x:"undistort_"+x , np.copy(self.filenames) ))
        images_list =  np.asarray(list( zip(self.images,self.undist_images) ))
        images_label =  np.asarray(list( zip(self.filenames,undistort_filenames) ))


    def makePipelineBinaryImage(self):

        """
        sobel_imagex = abs_sobel_thresh(x, orient='x', thresh_min=20, thresh_max=120, kernel_size=15)
        sobel_imagey = abs_sobel_thresh(x, orient='y', thresh_min=20, thresh_max=120, kernel_size=15)

        mag_binary = mag_thresh(x, sobel_kernel=15, mag_thresh=(80, 200))
        dir_binary = dir_threshold(x, sobel_kernel=15, thresh=(0.7, 1.3) )
        
        mybinary = np.zeros_like(dir_binary)
        mybinary[ (sobel_imagex == 1) | ( (mag_binary == 1) & (dir_binary == 1)      )      ] = 1
        """

        binary_images = []
        for im in self.undist_images:
            c, b = pipelineBinaryImage(im)
            binary_images.append(b)

        binary_labels = list(map( lambda x:"S channel + L channel Sobel binary "+x , self.filenames ))
        
        binary_images_label =  np.asarray(list( zip(self.filenames,binary_labels) ))
        binary_images_list =  np.asarray(list( zip(self.undist_images_gray, binary_images ) ))

        self.showImageOnDisplay(binary_images_list,binary_images_label,2)


def main():

    pipe = Pipeline()

    #pipe.displayNormalUndistort()
    pipe.makePipelineBinaryImage()


if __name__ == "__main__":
    main()

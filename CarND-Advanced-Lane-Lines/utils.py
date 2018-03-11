#
# Mar. 11th 
# H. Hamano
# the main purpose is to hold helper function of image binary 
# transformation
#

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


def threshold(img, thresh_min=0, thresh_max=255):
    # 
    # Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
    # 
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    
    return xbinary

#
# RGB Yellow + RGB White + HLS Yellow mixed binary image
#
def threshColoredImageBin(image):

    bin_thresh_min = 20
    bin_thresh_max = 255
    # rgb Yellow colored mask
    lower = np.array([255,180,0]).astype(np.uint8)
    upper = np.array([255,255,170]).astype(np.uint8)
    mask = cv2.inRange(image,lower,upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    y_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,y_binary,"Original Image","RGB Yellow filter",True)
    # rgb white colored mask
    lower = np.array([100,100,200]).astype(np.uint8)
    upper = np.array([255,255,255]).astype(np.uint8)
    mask = cv2.inRange(image,lower,upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    w_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,w_binary,"Original Image","RGB White filter",True)
    
    # HLS Yellow masking
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # rgb white colored mask
    lower = np.array([20,120,80]).astype(np.uint8)
    upper = np.array([45,200,255]).astype(np.uint8)
    mask = cv2.inRange(hls,lower,upper)
    hls_y = cv2.bitwise_and(image, image, mask=mask)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)    
    gray = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    y_hls_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,y_hls_binary,"Original Image","HLS Yellow filter",True)
    
    binary = np.zeros_like(y_binary)
    binary[ (y_binary == 1) | (w_binary == 1) | (y_hls_binary == 1) ] = 1

    return binary

#
# absolute soble thresh binary filtering image
#
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    if orient == 'x':
        yorder = 0
        xorder = 1
    else:
        yorder = 1
        xorder = 0    
    
    # Apply the following steps to img
    # 1) Convert to grayscale (RGB -> GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, xorder, yorder)
    # 3) Take the absolute value of the derivative or gradient
    sobel = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8( 255 * sobel / np.max(sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = threshold(sobel,thresh_min,thresh_max)
    #sxbinary = np.zeros_like(sobel)
    #sxbinary[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1
    return sxbinary
    
    # Run the function
    #grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    # Plot the result

#
#  magnitude of gradient proprocess
#
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # vertical and holizontal gradient 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    # magunitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel = np.sqrt( abs_sobelx **2 + abs_sobely ** 2  )
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8( 255 * sobel / np.max(sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    
    sxbinary = threshold(sobel,mag_thresh[0],mag_thresh[1])
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return sxbinary

#
# Direction of Gadient Image Process
#
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    # magunitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    #sobel = np.uint8( 255 * sobel / np.max(sobel))
    #print(sobel)
    # 5) Create a binary mask where mag thresholds are met
    sxybinary = np.zeros_like(sobel)
    sxybinary[(sobel > thresh[0]) & (sobel < thresh[1])] = 1    
    #binary_output = np.copy(img) # Remove this line
    return sxybinary
    

#
#  S channel of HLS binary filtering
# 

def hls_threshold(img, thresh=(0, 255)):
    
    # input is RGB (skimage.io.imread)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]  # 2 <-- S channel     
    
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary

def applyGradientCombination(image):
    
    ksize=3
    sobel_imagex = abs_sobel_thresh(image, orient='x', thresh_min=50, thresh_max=100)
    sobel_imagey = abs_sobel_thresh(image, orient='y', thresh_min=50, thresh_max=100)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    mybinary = np.zeros_like(dir_binary)
    mybinary[ ( (sobel_imagex == 1) & (sobel_imagey == 1)) | ( (mag_binary == 1) & (dir_binary == 1) ) ] = 1

    return mybinary

#
#   sobelx binary + s_channel colored filtering
#   Green channel = soblel binary filtering
#   Blue channel = S channel of HLS image.
#  
def pipelineColorImage(x):

    sobelx_image = abs_sobel_thresh(x, orient='x', thresh_min=20, thresh_max=100)
    s_channel_image = hls_threshold(x, thresh=(170,255))
    color_binary = np.dstack(( np.zeros_like(sobelx_image), sobelx_image, s_channel_image)) * 255    
    # 
    # color_binary is 3 channel.
    #
    return color_binary

#
#   sobelx binary + s_channel binary filtering
#
def pipelineBinaryImage(x):

    s_channel_image = hls_threshold(x, thresh=(170,255))
    sobelx_image = abs_sobel_thresh(x, orient='x', thresh_min=20, thresh_max=100)

    combined_binary = np.zeros_like(s_channel_image)
    combined_binary[ (sobelx_image == 1)  | ( s_channel_image == 1) ] = 1

    #
    # binary image
    #
    return combined_binary

#
#  l channel of HLS applied to sobelx  
#  S channel of HLS binary filtering. 
def pipelineBinaryImage2(image, s_thresh=(170, 255), sx_thresh=(20, 100)):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_image = hls[:,:,2]
    l_image = hls[:,:,1]

    sobelx = cv2.Sobel(l_image, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    #plt.imshow(sxbinary)
    
    # Threshold color channel
    s_binary = np.zeros_like(s_image)
    s_binary[(s_image >= s_thresh[0]) & (s_image <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary   )) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    #
    # result is 2 output parameters 
    # a. colored binary = 3 channels 
    # b. combined parameters. binary image
    #
    return color_binary, combined_binary
    
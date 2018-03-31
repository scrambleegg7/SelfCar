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
    lower = np.array([0,100,100]).astype(np.uint8)
    upper = np.array([50,255,255]).astype(np.uint8)
    mask = cv2.inRange(image,lower,upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    y_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,y_binary,"Original Image","RGB Yellow filter",True)
    # rgb white colored mask
    lower = np.array([18,0,180]).astype(np.uint8)
    upper = np.array([255,80,255]).astype(np.uint8)
    mask = cv2.inRange(image,lower,upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    w_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,w_binary,"Original Image","RGB White filter",True)
    
    # HLS Yellow masking
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # rgb white colored mask
    lower = np.array([18,0,180]).astype(np.uint8)
    upper = np.array([255,80,255]).astype(np.uint8)
    mask = cv2.inRange(hls,lower,upper)
    hls_y = cv2.bitwise_and(image, image, mask=mask)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)    
    gray = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    y_hls_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    #displayImage2x1(image,y_hls_binary,"Original Image","HLS Yellow filter",True)
    
    binary = np.zeros_like(y_binary)
    binary[ (y_binary == 1) | (w_binary == 1) | (y_hls_binary == 1) ] = 1

    return (binary,y_binary,w_binary,y_hls_binary)

#
# absolute soble thresh binary filtering image
#
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, kernel_size=3):

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
    sobelx = cv2.Sobel(gray, cv2.CV_64F, xorder, yorder, ksize = kernel_size)
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

def applyCombinedGradient(x):
    
    sobel_imagex = abs_sobel_thresh(x, orient='x', thresh_min=20, thresh_max=120, kernel_size=15)
    sobel_imagey = abs_sobel_thresh(x, orient='y', thresh_min=20, thresh_max=120, kernel_size=15)

    mag_binary = mag_thresh(x, sobel_kernel=15, mag_thresh=(80, 200))
    #dir_binary = dir_threshold(x, sobel_kernel=15, thresh=(np.pi/4, np.pi/2) )
    dir_binary = dir_threshold(x, sobel_kernel=15, thresh=(0.7, 1.3) )
    dir_binary = dir_binary.astype(np.uint8)
    
    mybinary = np.zeros_like(dir_binary)
    mybinary[(sobel_imagex == 1) |  ((sobel_imagey == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1
    # mybinary[ (sobel_imagex == 1) | ( (mag_binary == 1) & (dir_binary == 1)      )      ] = 1
    
    return mybinary

def pipelineBinaryImage(image):

    # incoming is undistortion image 

    combined_image = applyCombinedGradient(image)
    hls_image = hls_threshold(image, thresh=(170, 255))
            
    combined_binary = np.zeros_like(combined_image)
    combined_binary[ (combined_image == 1)  | ( hls_image == 1) ] = 1

    color_binary = np.dstack(( np.zeros_like(combined_image), combined_image, hls_image   )) * 255
    color_binary = color_binary.astype(np.uint8)
    
    return color_binary, combined_binary
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
def pipelineBinaryImage2(image, s_thresh=(170, 255), sx_thresh=(20, 100)):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_image = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_image)
    s_binary[(s_image >= s_thresh[0]) & (s_image <= s_thresh[1])] = 1

    sobel_imagex = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=120, kernel_size=15)
    sobel_imagey = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=120, kernel_size=15)
    mag_binary = mag_thresh(image, sobel_kernel=15, mag_thresh=(80, 200))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(np.pi/4, np.pi/2) )

    sxbinary = np.zeros_like(dir_binary)
    #mybinary[ (sobel_imagex == 1)  | ( (mag_binary == 1) & (dir_binary == 1)      )      ] = 1
    sxbinary[(sobel_imagex == 1) |  ((sobel_imagey == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary   )) * 255
    color_binary = color_binary.astype(np.uint8)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    #
    # result is 2 output parameters 
    # a. colored binary = 3 channels 
    # b. combined parameters. binary image
    #
    return color_binary, combined_binary#
#  l channel of HLS applied to sobelx  
#  S channel of HLS binary filtering. 
def pipelineBinaryImage3(image, s_thresh=(170, 255), sx_thresh=(20, 100)):

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

def pipelineBinaryImage4(x, s_thresh=(170, 255), sx_thresh=(20, 100)):

    hls = cv2.cvtColor(x, cv2.COLOR_RGB2HLS)
    s_image = hls[:,:,2]
    l_image = hls[:,:,1]

    sobelx = cv2.Sobel(l_image, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    sobely = cv2.Sobel(l_image, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))    
    # Threshold x gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sx_thresh[0]) & (scaled_sobely <= sx_thresh[1])] = 1

    sxybinary = np.zeros_like(scaled_sobel)
    sxybinary[ (sxbinary == 1)  & ( sybinary == 1) ] = 1

    s_binary = np.zeros_like(s_image)    
    s_binary[(s_image > s_thresh[0]) & (s_image <= s_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxybinary), sxybinary, s_binary   )) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxybinary == 1)] = 1
    
    #
    # result is 2 output parameters 
    # a. colored binary = 3 channels 
    # b. combined parameters. binary image
    #
    return color_binary, combined_binary



def showImageList(images_list, images_label, cols=2, fig_size=(26, 22) ):

    rows = len(images_list)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(hspace=0.3,wspace=0.07)
    fig1 = plt.figure(figsize=fig_size)
    ax = []
    cmap = None
    for i in range( rows * cols):        
        r = (i // cols)
        c = i % cols

        img = images_list[r][c]
        lbl = images_label[r][c]
        
        if len(img.shape) < 3 or img.shape[-1] < 3:
            cmap = "gray"
            img = np.reshape(img, (img.shape[0], img.shape[1]))        
        
        ax.append(fig1.add_subplot(gs[r, c]))
        ax[-1].set_title('%s' % str(lbl))
        
        ax[-1].imshow(img, aspect="auto", cmap=cmap)
        #ax[-1].axis("off")
    plt.show()
    

def undistort_corners_unwarp(img, src, mtx, dist):

    # images should be undistortion based on camera calibration.
    # incoming image is RGB format --> skimage.io.imread
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    
    # Define 4 destination points
    
    # Mar. 10, 2018 
    # dst is selected after a number of experimental trial to find
    # what is best destination combination to show 
    # straight line of the warped image

    # destination values are hardcoded as follows; 
    dst = np.float32([[250, img.shape[0]], [250, 0], 
                      [960, 0], [960, img.shape[0]]])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv


def equYCrCb(img):

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    ch1 = ycrcb[:,:,0]
    ch2 = ycrcb[:,:,1]
    ch3 = ycrcb[:,:,2]

    # apply histogram equalization for each channel
    equ1 = cv2.equalizeHist(ch1)
    equ2 = ch2
    equ3 = ch3

    # stack/combine channels again
    histeq = np.dstack((equ1,equ2,equ3))
    rgb = cv2.cvtColor(histeq, cv2.COLOR_YCrCb2RGB)
    
    return rgb

def ycrcbthresh(img,thr=(230, 255)):

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    ch1 = ycrcb[:,:,0]
    ch2 = ycrcb[:,:,1]
    ch3 = ycrcb[:,:,2]
    ch1 = cv2.equalizeHist(ch1)

    binary1 = threshold(ch1, thresh_min=thr[0], thresh_max=thr[1])
    binary2 = threshold(ch2, thresh_min=thr[0], thresh_max=thr[1])    
    binary3 = threshold(ch3, thresh_min=thr[0], thresh_max=thr[1])

    return binary1, binary2, binary3
    

def rgbthresh(img,thr=(230, 255)):

    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    binary1 = threshold(ch1, thresh_min=thr[0], thresh_max=thr[1])
    binary2 = threshold(ch2, thresh_min=thr[0], thresh_max=thr[1])    
    binary3 = threshold(ch3, thresh_min=thr[0], thresh_max=thr[1])

    return binary1, binary2, binary3

def hsvthresh(img,thr=(230, 255)):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]   
    binary1 = threshold(ch1, thresh_min=thr[0], thresh_max=thr[1])
    binary2 = threshold(ch2, thresh_min=thr[0], thresh_max=thr[1])    
    binary3 = threshold(ch3, thresh_min=thr[0], thresh_max=thr[1])

    return binary1, binary2, binary3

def luvthresh(img,thr=(157, 255)):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]   
    binary1 = threshold(ch1, thresh_min=thr[0], thresh_max=thr[1])
    binary2 = threshold(ch2, thresh_min=thr[0], thresh_max=thr[1])    
    binary3 = threshold(ch3, thresh_min=thr[0], thresh_max=thr[1])

    return binary1, binary2, binary3

def buildYBinary(img, thr_rgb=(230,255), thr_hsv=(230,255), thr_luv=(157,255)):

    rbin1, rbin2, rbin3 = rgbthresh(img,thr_rgb)
    hbin1, hbin2, hbin3 = hsvthresh(img,thr_hsv)
    lbin1, lbin2, lbin3 = luvthresh(img,thr_luv)
    
    ybin1, ybin2, ybin3 = ycrcbthresh(img,thr_rgb)


    binary = np.zeros_like(rbin1)
    binary[
        #(rbin1 == 1)
        (ybin1 == 1)
        | (hbin3 == 1)
        | (lbin3 == 1)
        ] = 1
    
    return binary

def buildBinary(img, thr_rgb=(230,255), thr_hsv=(230,255), thr_luv=(157,255)):

    rbin1, rbin2, rbin3 = rgbthresh(img,thr_rgb)
    hbin1, hbin2, hbin3 = hsvthresh(img,thr_hsv)
    lbin1, lbin2, lbin3 = luvthresh(img,thr_luv)
    
    ybin1, ybin2, ybin3 = ycrcbthresh(img,thr_rgb)


    binary = np.zeros_like(rbin1)
    binary[
        (rbin1 == 1)
        | (hbin3 == 1)
        | (lbin3 == 1)
        ] = 1
    
    return binary

def HLS_S_thresh(img, thr = (80,200)):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ch1 = img[:,:,0]   # H
    ch2 = img[:,:,1]   # L
    ch3 = img[:,:,2]   # S
    binary1 = threshold(ch3,thr[0],thr[1])

    return binary1

def HSV_V_SobelX_thresh(img, thr = (20,100) ):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ch1 = img[:,:,0]    # H
    ch2 = img[:,:,1]    # S
    ch3 = img[:,:,2]    # V

    yorder = 0
    xorder = 1

    sobelx = cv2.Sobel(ch3, cv2.CV_64F, xorder, yorder, ksize = 15)
    sobel = np.absolute(sobelx)
    sobel = np.uint8( 255 * sobel / np.max(sobel))
    binary2 = threshold(sobel,thr[0],thr[1])
    
    return binary2

def makeLOWCONTRAST(img, thr_HLS=(80,200), thr_HSV=(20,100) ):

    binary1 = HLS_S_thresh(img, thr_HLS)
    binary2 = HSV_V_SobelX_thresh(img, thr_HSV)

    binLowContrast = np.zeros_like(binary1)
    binLowContrast[(binary1 == 1) | (binary2 == 1)] = 1
    
    return binLowContrast

def birds_eye_transform_challenge(img,  offsetx):
    """Transforms the viewpoint to a bird's-eye view.
    Args:
        img: A numpy image array.
        points: A list of four points to be flattened.
            Example: points = [[x1,y1], [x2,y2], [x4,y4], [x3,y3]].
        offsetx: offset value for x-axis.
    """

    # revert of shape format shape[1] shape[0]
    img_size = img[:,:,0].shape[::-1]
    
    #pt1 = [offsetx, 0]
    #pt2 = [img_size[0] - offsetx, 0]
    #pt3 = [img_size[0] - offsetx, img_size[1]]
    #pt4 = [offsetx, img_size[1]]
    #dst = np.float32([pt1, pt2, pt3, pt4])
    
    src = np.float32([[585, 450], [203, 720], [1127, 720], [695, 450]])
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])    
    mtx = cv2.getPerspectiveTransform(src, dst)
    invmtx = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, mtx, img_size)
    
    return invmtx, warped

def color_thresholding(
    img, ch_type='rgb', 
    binary=True, plot=False, 
    thr=(220, 255), save_path=None):
    """Apply color thresholding.
    
    Arg:
        img (numpy array): numpy image array, should be in `RGB`
            color space, NOT in `BGR`.
            
        ch_type (str): can be 'rgb', 'hls', 'hsv', 'yuv', 'ycrcb',
            'lab', 'luv'.
            
        binary (bool): If `True` then show and returns binary
            images. If not, returns original images in defined 
            color spaces.
            
        plot: If `True`, shows images.
        
        thr: min, max value for threasholding.
        
        save_path: if defines, saves figures.
    """
    # get channels
    if ch_type is 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif ch_type is 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif ch_type is 'yuv':    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif ch_type is 'ycrcb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif ch_type is 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    elif ch_type is 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    
    img_ch1 = img[:,:,0]
    img_ch2 = img[:,:,1]
    img_ch3 = img[:,:,2]

    # apply thresholding
    bin_ch1 = np.zeros_like(img_ch1)
    bin_ch2 = np.zeros_like(img_ch2)
    bin_ch3 = np.zeros_like(img_ch3)

    bin_ch1[(img_ch1 > thr[0]) & (img_ch1 <= thr[1])] = 1
    bin_ch2[(img_ch2 > thr[0]) & (img_ch2 <= thr[1])] = 1
    bin_ch3[(img_ch3 > thr[0]) & (img_ch3 <= thr[1])] = 1
    
    if binary:
        imrep_ch1 = bin_ch1
        imrep_ch2 = bin_ch2
        imrep_ch3 = bin_ch3
    else:
        imrep_ch1 = img_ch1
        imrep_ch2 = img_ch2
        imrep_ch3 = img_ch3
        
    return imrep_ch1, imrep_ch2, imrep_ch3

def combined_color_thresholding(
    img, thr_rgb=(230,255), thr_hsv=(230,255), thr_luv=(157,255)):
    """Combines color thresholding on different channels
    
    Returns a binary image.
    
    Args:
        img: Numpy image array, should be in `RGB` color
            space.
        
        thr_rgb: min and max thresholding values for RGB color
            space.
        
        thr_hsv: min and max thresholding values for HSV color
            space.
        
        thr_luv: min and max thresholding values for LUV color
            space.
    """
    
    bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 =\
        color_thresholding(img, ch_type='rgb', thr=thr_rgb)
        
    bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 =\
    color_thresholding(img, ch_type='hsv', thr=thr_hsv)
    
    bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 =\
    color_thresholding(img, ch_type='luv', thr=thr_luv)
    
    binary = np.zeros_like(bin_rgb_ch1)
    binary[
        (bin_rgb_ch1 == 1)
        | (bin_hsv_ch3 == 1)
        | (bin_luv_ch3 == 1)
        ] = 1
    
    return binary
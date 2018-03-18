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

from utils import hls_threshold
from utils import applyCombinedGradient

from utils import mag_thresh, dir_threshold

from utils import showImageList
from utils import pipelineBinaryImage
from utils import pipelineBinaryImage2
from utils import pipelineBinaryImage3
from utils import pipelineBinaryImage4

import platform

from Pipeline import Pipeline
from utils import pipelineBinaryImage

class LaneDetector(object):

    def __init__(self):

        # pipeline encupsule necessary parameters for image transformation.
        # mtx / dist are saved initial process of Pipeline.
        self.mypipeline = Pipeline()

        self.setSrcDst()

        # was the line detected in the last iteration?
        self.detected = False  

    def setSrcDst(self):

        self.src = np.float32([[270,674],
                 [579,460],
                 [702,460],
                 [1060,674]])
        self.dst = np.float32([[270,674],
                 [270,0],
                 [1035,0],
                 [1035,674]])

    def M_gen(self,src,dst):
    
        # Given src and dst points, calculate the perspective transform matrix
        # source is 4 points trapezoid
        # destination is 4 points rectangle
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def warp_gen(self,img,M):
        # check out the image shape
        img_size = ( img.shape[1], img.shape[0] )
        warped = cv2.warpPerspective(img, M, img_size)
        # Return the resulting image and matrix
        return warped

    def undistort_corners(self,img, mtx, dist):

        # images should be undistortion based on camera calibration.
        # incoming image is RGB format --> skimage.io.imread
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    #
    # -------------- mathematical function
    # 
    def formula2(self,a, b, c, y):
        
        x = a * y ** 2 + b * y + c
        return x 

    def fitting(self, lefty,leftx,righty,rightx):

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit,right_fit

    def fitXY(self, ploty,left_fit,right_fit):
        
        #print("fitXY - left fit:",left_fit)
        #print("fitXY - right fit:",right_fit)    
        
        #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        a, b, c = left_fit
        left_fitx = self.formula2(a, b, c, ploty)
        
        a, b, c = right_fit
        right_fitx = self.formula2(a, b, c, ploty)
        
        return left_fitx,right_fitx

    def curvature(self, ploty, left_fitx, right_fitx):

        # 
        # This logic can be used for sanitary check polyfit calculation.
        #
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720.0 # meters per pixel in y dimension
        xm_per_pix = 3.7/700.0 # meters per pixel in x dimension
        
        lefty = ploty*ym_per_pix
        leftx = left_fitx*xm_per_pix
        righty = ploty*ym_per_pix
        rightx = right_fitx*xm_per_pix
        
        left_fit_cr, right_fit_cr = self.fitting(lefty,leftx,righty,rightx)
        #left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        #right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)    
        # Calculate the new radii of curvature
        a, b, c = left_fit_cr
        left_curverad = ((1 + (2*a*y_eval*ym_per_pix + b)**2)**1.5) / np.absolute(2*a)
        
        a, b, c = right_fit_cr
        right_curverad = ((1 + (2*a*y_eval*ym_per_pix + b)**2)**1.5) / np.absolute(2*a)

        return left_curverad, right_curverad

    #
    # --------------------------------------------------------------
    #

    def makeWarpBinaryImage(self,image):
        
        mtx = self.mypipeline.mtx
        dist = self.mypipeline.dist

        # call perspective to get M / Minv
        self.M, self.Minv = self.M_gen(self.src,self.dst)
        # incoming image converted to undistortion
        undist = self.undistort_corners(image, mtx, dist)
        # make warp image
        self.warp_image = self.warp_gen(undist,self.M)
        
        c, b = pipelineBinaryImage4(self.warp_image)

        return c, b

    def displyIntegrateView(self, out_img,ploty,left_fitx,right_fitx):

        plt.figure(figsize=(10,8))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        #plt.title("Curved Line1",fontsize="22")
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    def showInitialfittingPlotXY(self):

        rows = 4
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(15,13 ) )

        images = self.mypipeline.images
        filenames = self.mypipeline.filenames

        for ax, im, filename in zip(axes.flat, images, filenames):
            
            self.test(ax,im,filename)

        plt.show()
    #
    # ------------ Pipeline process
    #
    def image_pipeline(self,x):

        #curvature(ploty, left_fitx, right_fitx)        
        ploty = np.linspace(0, x.shape[0]-1, x.shape[0] )
        
        # generate Binary Warped Image from incoming normal image
        _, bw = self.makeWarpBinaryImage(x)
 
        out_img,midpoint,leftx_base,rightx_base,leftx,lefty,rightx,righty = self.searchinglaneline(bw)
 
    def genResultImageAfterFitting(self,binary_warped, ploty, left_fit,right_fit):

        #
        # left_fit & right_fit ==> calculated with fitting method
        #
        
        leftx,lefty,rightx,righty = foundlaneline(binary_warped,left_fit,right_fit)
        left_fit_new,right_fit_new = fitting(lefty,leftx,righty,rightx)
        left_fitx,right_fitx = fitXY(ploty,left_fit_new,right_fit_new)
        
        #
        # This is called after fitting parameters searched with fitting function
        #
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
            
        out_img[lefty,leftx] = [255,0,0] # LEFT RED
        out_img[righty,rightx] = [0,0,255] # RIGHT BLUE
        
        # Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        margin = 100
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        #displyIntegrateView(result,ploty,left_fitx,right_fitx)

        return result, left_fitx, right_fitx


    
    def test(self,ax, x, filename):

        fontScale=1
        #undist_images = self.mypipeline.undist_images
        
        #
        # Vertical lines from bottom to top is defined.
        # this is used to plot predicatable line.
        #
        
        ploty = np.linspace(0, x.shape[0]-1, x.shape[0] )


        # generate Binary Warped Image
        _, bw = self.makeWarpBinaryImage(x)

        out_img,midpoint,leftx_base,rightx_base,leftx,lefty,rightx,righty = self.searchinglaneline(bw)

        left_fit,right_fit = self.fitting(lefty,leftx,righty,rightx)
        left_fitx,right_fitx = self.fitXY(ploty,left_fit,right_fit)


        init_left_curved, init_right_curved = self.curvature(ploty, left_fitx, right_fitx)
        
        # Draw info for fitting parameters 
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 255, 255)
        cv2.putText(out_img, 'Left fitting -> a: {:.4f}'.format(left_fit[0]), (50, 50), font, fontScale, fontColor, 2)
        cv2.putText(out_img, 'b: {:.4f}'.format(left_fit[1]), (500, 50), font, fontScale, fontColor, 2)
        cv2.putText(out_img, 'c: {:.4f}'.format(left_fit[2]), (780, 50), font, fontScale, fontColor, 2)
        
        cv2.putText(out_img, 'Right fitting -> : {:.4f} '.format(right_fit[0]), (50, 100), font, fontScale, fontColor, 2)
        cv2.putText(out_img, 'b: {:.4f}'.format(right_fit[1]), (500, 100), font, fontScale, fontColor, 2)
        cv2.putText(out_img, 'c: {:.4f}'.format(right_fit[2]), (780, 100), font, fontScale, fontColor, 2)
        

        #cv2.putText(out_img, 'Vehicle is {} of center'.format(message), (50, 190), font, fontScale, fontColor, 2)
        #self.displyIntegrateView(out_img,ploty,left_fitx,right_fitx)
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')

        ax.set_title(filename)
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)


        #plt.imshow(out_img)
        #plt.show()

    def searchinglaneline(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        #print(histogram)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        #
        # Set height of windows = 720 / 9 = 80px
        # It should be constant if we read video format.
        #
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        #print("window hight --> %spx " %window_height)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base    
        
        # Set the width of the windows +/- margin
        margin = 100 # px
        # Set minimum number of pixels found to recenter window
        minpix = 50 # 
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        
        for i, window in enumerate(range(nwindows)):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # for debug
            #print(i,  win_xleft_low,win_xleft_high,win_xright_low,win_xright_high, win_y_low, win_y_high)
            # Draw the windows on the visualization image
            GREEN_C = (0,255,0)
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),GREEN_C, 4) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),GREEN_C, 4) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        # Concatenate the arrays of indices to make flat array
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        

        return out_img,midpoint,leftx_base,rightx_base,leftx,lefty,rightx,righty


def main():

    lane_detector = LaneDetector()

    lane_detector.showInitialfittingPlotXY()


if __name__ == "__main__":
    main()

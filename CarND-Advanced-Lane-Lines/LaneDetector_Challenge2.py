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

from utils_challenge import *

import platform

from Pipeline import Pipeline

class LaneDetector(object):

    def __init__(self):

        # pipeline encupsule necessary parameters for image transformation.
        # mtx / dist are saved initial process of Pipeline.
        self.mypipeline = Pipeline()

        self.setSrcDst()

        # was the line detected in the last iteration?
        self.detected = False  

        # for saving fitting parameters 
        self.left_fit_array = []
        self.right_fit_array = []
        # calculated left_fit
        self.best_left_fit = None 
        # calculated right_fit
        self.best_right_fit = None

        #
        self.left_outlier = False
        #
        self.right_outlier = False

        #
        self.current_left_fit = [np.array([False])]  
        self.current_right_fit = [np.array([False])]  

        self.good_curve_left_fit = []
        self.good_curve_right_fit = []
        
        # left curved
        self.left_curved = None
        # right curved
        self.right_curved = None

        self.initial_frame = True

    def setSrcDst(self):

        self.src = np.float32([[270,674],
                 [579,460],
                 [702,460],
                 [1060,674]])
        self.dst = np.float32([[270,674],
                 [270,0],
                 [1035,0],
                 [1035,674]])

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


        # for your reference to calculate center of the car positions.
        #https://discussions.udacity.com/t/where-is-the-vehicle-in-relation-to-the-center-of-the-road/237424/2

        camera_position = self.undist.shape[1]/2
        lane_center = (left_fitx[-1] + right_fitx[-1])/2
        center_offset_pixels = abs(camera_position - lane_center)
        dx = center_offset_pixels * xm_per_pix

        #centerOfLanes = (left_fitx[-1] + right_fitx[-1])//2
        #veh_pos = self.warp_image.shape[0]//2
        #dx = (veh_pos - centerOfLanes)*xm_per_pix # Positive if on right, Negative on left        
        #centerOfLanes = (self.leftx_base + self.rightx_base) / 2
        #offset = (centerOfLanes-( self.warp_image.shape[0]  / 2))*xm_per_pix

        return left_curverad, right_curverad, dx

    #
    # --------------------------------------------------------------
    #
    #
    # ------------ Pipeline process
    #
    def makeWarpImage(self,image):
        
        mtx = self.mypipeline.mtx
        dist = self.mypipeline.dist

        img_size = np.shape(image)

        ht_window = np.uint(img_size[0]/1.5)
        hb_window = np.uint(img_size[0])
        c_window = np.uint(img_size[1]/2)
        ctl_window = c_window - .2*np.uint(img_size[1]/2)
        ctr_window = c_window + .2*np.uint(img_size[1]/2)
        cbl_window = c_window - .9*np.uint(img_size[1]/2)
        cbr_window = c_window + .9*np.uint(img_size[1]/2)

        src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])

        dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],
                        [img_size[1],0],[0,0]])
        
        warped, self.M, self.Minv = warp_image(image,src,dst,(img_size[1],img_size[0]))

        self.warp_image = warped.copy()

        return warped

    def makeBinaryImage2(self, warped):

        hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
        lower = np.array([20,60,60])
        upper = np.array([38,174, 250])
        mask_yellow = cv2.inRange(hsv, lower, upper)        

        lower = np.array([202,202,202])
        upper = np.array([255,255,255])
        mask_white = cv2.inRange(warped, lower, upper)

        combined_binary = np.zeros_like(mask_yellow)
        combined_binary[(mask_yellow >= 1) | (mask_white >= 1)] = 1

        return combined_binary

    def makeBinaryImage(self,warped):
        # yellow mask
        image_HSV = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)
        yellow_hsv_low  = np.array([ 0,  100,  100])
        yellow_hsv_high = np.array([ 80, 255, 255])    
        #res = apply_color_mask(image_HSV,warped,yellow_hsv_low,yellow_hsv_high)

        # white color mask
        image_HSV = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)
        #white_hsv_low  = np.array([ 0,   0,   160])
        #white_hsv_high = np.array([ 255,  80, 255])
        
        white_hsv_low  = np.array([  0,   0,    200])
        white_hsv_high = np.array([ 255,  80, 255])

        #res1 = apply_color_mask(image_HSV,warped,white_hsv_low,white_hsv_high)

        mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
        mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
        mask_lane = cv2.bitwise_or(mask_yellow,mask_white)

        return mask_lane
    
    def sobelFilter(self, warped):

        #image = gaussian_blur(warped, kernel=5)
        image_HLS = cv2.cvtColor(warped,cv2.COLOR_RGB2HLS)

        img_gs = image_HLS[:,:,1]
        img_abs_x = abs_sobel_thresh(img_gs,'x',15,(50,225))
        img_abs_y = abs_sobel_thresh(img_gs,'y',15,(50,225))
        
        wraped2 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

        img_gs = image_HLS[:,:,2]
        img_abs_x = abs_sobel_thresh(img_gs,'x',15,(50,255))
        img_abs_y = abs_sobel_thresh(img_gs,'y',15,(50,255))

        wraped3 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

        image_cmb = cv2.bitwise_or(wraped2,wraped3)
        
        return image_cmb

    def combinedBinaryImage(self, image):
        
        mask_lane = self.makeBinaryImage2(image)
        #image_cmb = self.sobelFilter(image)
        #image_cmb1 = np.zeros_like(image_cmb)
        #image_cmb1[(mask_lane>=.5)|(image_cmb>=.5)]=1

        return mask_lane


    def histogramSearch(self,binary_warped):

        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0) 
        #shistogram = np.sum(binLowContrast[  binLowContrast.shape[0]//2:,:  ], axis=0)
        
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        #
        # midpoint = center of the road
        # leftx_base = most left point of lane line detected 
        # rightx_base = most right point of lane line detected
        #  
        return midpoint, leftx_base, rightx_base

    
    def lane_search(self):
        pass

    def line_detector(self,binary_warped):

        midpoint, leftx_base, rightx_base = self.histogramSearch(binary_warped)

        self.leftx_base = leftx_base
        self.rightx_base = rightx_base

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

        total_non_zeros = len(nonzeroy)
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base    
        
        # Set the width of the windows +/- margin
        margin = 100 # px
        # Set minimum number of pixels found to recenter window
        minpix = 40 # 
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        #self.detected = False
        if self.detected == False:
            #
            # sliding window technique

            # for statistical bucketing analysis
            leftx_means = []
            leftx_stds = []
            rightx_means = []
            rightx_stds = []
            self.left_outlier = False
            self.right_outlier = False

            #
            for i, window in enumerate(range(nwindows)):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                
                # -----------------------
                # left side
                #
                if len(good_left_inds) > minpix:

                    good_leftx = nonzerox[good_left_inds]
                    leftx_mean = np.mean(good_leftx)
                    leftx_std = np.std(good_leftx)
                    
                    leftx_means.append(leftx_mean)
                    leftx_stds.append(leftx_std)
                    
                    #
                    # stdev calculation if any wide stdev is given, the last item will be deleted. threshold > 0.5
                    #
                    leftx_stds_moving_ratio = ( leftx_std - np.mean(leftx_stds[:i+1]) ) / np.mean(leftx_stds[:i+1])
                    #print("left diffs from piled stds -->",  leftx_stds_moving_ratio )
                    
                    if abs(leftx_stds_moving_ratio) > .5:
                        del left_lane_inds[-1]
                        self.left_outlier = True
                    


                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

                # ---------------------
                # right side
                #     
                if len(good_right_inds) > minpix:


                    good_rightx = nonzerox[good_right_inds]
                    rightx_mean = np.mean(good_rightx)
                    rightx_std = np.std(good_rightx)
                    
                    rightx_means.append(rightx_mean)
                    rightx_stds.append(rightx_std)
                    
                    #
                    #  right side std calculation - if any wide std is given, last index will be deleted.
                    #
                    
                    rightx_stds_moving_ratio = ( rightx_std - np.mean(rightx_stds[:i+1]) )  / np.mean(rightx_stds[:i+1])
                    #print("right diffs from piled means -->", ( rightx_mean - np.mean(rightx_means[:i+1]) ) / np.mean(rightx_means[:i+1])    )
                    #print("right diffs from piled stds -->",  rightx_stds_moving_ratio     )
                    if abs(rightx_stds_moving_ratio) > 0.5:
                        del right_lane_inds[-1]
                        self.right_outlier = True

                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            length_left_lane_ids = len(left_lane_inds) 
            length_right_lane_ids = len(right_lane_inds)

            # Concatenate the arrays of indices to make flat array
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # for initial setup 
            if length_left_lane_ids > 5 and length_right_lane_ids > 5:

                leftx_init = nonzerox[left_lane_inds]
                lefty_init = nonzeroy[left_lane_inds] 
                rightx_init = nonzerox[right_lane_inds]
                righty_init = nonzeroy[right_lane_inds]

                left_fit = np.polyfit(lefty_init, leftx_init, 2)
                right_fit = np.polyfit(righty_init, rightx_init, 2)

                self.good_curve_left_fit.append( left_fit   )
                self.good_curve_right_fit.append(  right_fit  )

                # best fit parameters LEFT
                self.best_left_fit = np.mean(  self.good_curve_left_fit[-10:], axis=0)  
                self.best_right_fit = np.mean(  self.good_curve_right_fit[-10:], axis=0)  


            if self.left_outlier or self.right_outlier:
                self.detected = False
            else:
                self.detected = True

        else: 

            left_lane_inds = ((nonzerox > (self.best_left_fit[0]*(nonzeroy**2) + self.best_left_fit[1]*nonzeroy + \
                            self.best_left_fit[2] - margin)) & (nonzerox < (self.best_left_fit[0]*(nonzeroy**2) + \
                            self.best_left_fit[1]*nonzeroy + self.best_left_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (self.best_right_fit[0]*(nonzeroy**2) + self.best_right_fit[1]*nonzeroy + \
                            self.best_right_fit[2] - margin)) & (nonzerox < (self.best_right_fit[0]*(nonzeroy**2) + \
                            self.best_right_fit[1]*nonzeroy + self.best_right_fit[2] + margin)))  





        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        #print("length left right x y ", len(leftx), len(lefty), len(rightx), len(righty)  )
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )        

        if len(leftx) < 100 or len(lefty) < 100:
            left_fit = self.best_left_fit
        else:    
            left_fit = np.polyfit(lefty, leftx, 2)
        
        if len(rightx) < 100 or len(righty) < 100:
            right_fit = self.best_right_fit
        else:
            right_fit = np.polyfit(righty, rightx, 2)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        x_std_left = np.std( leftx )
        x_std_right = np.std( rightx )

        x_std_left_fitx = np.std( left_fitx )
        x_std_right_fitx = np.std( right_fitx )


        #print( x_std_left, x_std_right, x_std_left_fitx, x_std_right_fitx  )

        if x_std_left > 50. or x_std_right > 50.:
            self.detected = False
            print(" out of standard deviation " , x_std_left, x_std_right)

        #plt.imshow(out_img)                
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()


        #
        # curvature calculation
        #
        init_left_curved, init_right_curved, off_center = self.curvature(ploty, left_fitx, right_fitx)

        #print("curvature :",  init_left_curved,init_right_curved)
        if (init_left_curved > 300 and init_left_curved < 2000):
            self.good_curve_left_fit.append( left_fit   )
            left_fit = np.mean(  self.good_curve_left_fit[-10:], axis=0)  
            self.best_left_fit = left_fit
        else:
            left_fit = self.best_left_fit
            self.detected = False
            print(" left lane out of range! ")

        if (init_right_curved > 300 and init_right_curved < 2000):
            self.good_curve_right_fit.append(  right_fit  )
            right_fit = np.mean(  self.good_curve_right_fit[-10:], axis=0)  
            self.best_right_fit = right_fit
        else:
            right_fit = self.best_right_fit
            self.detected = False
            print(" right lane out of range !")


        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        init_left_curved, init_right_curved, off_center = self.curvature(ploty, left_fitx, right_fitx)
    
        fontScale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 255, 255)
        fontColorR = (255, 0, 0)

        cv2.putText(self.undist, 'Left Curvature  : {:.2f} m'.format(init_left_curved), (50, 50), font, fontScale, fontColorR, 2)
        cv2.putText(self.undist, 'Right Curvature  : {:.2f} m'.format(init_right_curved), (50, 100), font, fontScale, fontColorR, 2)
        cv2.putText(self.undist, 'Off Center Position : {:.2f} m'.format(off_center), (50, 200), font, fontScale, fontColorR, 2)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        img_size = binary_warped.shape[::-1]
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, img_size ) 
        # Combine the result with the original image
        result = cv2.addWeighted(self.undist, 1, newwarp, 0.3, 0)
        #plt.imshow(result)
        #plt.show()
        self.initial_frame = False

        return result

    def image_pipeline(self,x):

        mtx = self.mypipeline.mtx
        dist = self.mypipeline.dist
        undist_ops = lambda img:cv2.undistort(img, mtx, dist, None, mtx)
        self.undist = undist_ops(x)

        warped = self.makeWarpImage(self.undist)
        #binLowContrast = self.makeBinaryImage(warped)
        binLowContrast = self.combinedBinaryImage(warped)
        
        # line_detector accepts the warped binary image 
        res = self.line_detector(binLowContrast)

        return res

def main():

    lane_detector = LaneDetector()

    #filenames = sorted( os.listdir("./challenge") )
    imread_ops = lambda x: imread( os.path.join("./project",x) )
    #undist_ops = lambda img:cv2.undistort(img, mtx, dist, None, mtx)

    filename_ops = lambda r:'frame{:d}.jpg'.format(r)
    images = {}
    for  r in range(1259):
        images[r] = imread_ops(filename_ops( r )  )

    #filenames = sorted( images.keys() )
    for idx, (index, image) in enumerate( sorted( images.items(), key=lambda x:x[0]  ) ): 

        print("image index : ", filename_ops(idx) , image.shape )

        test_result = lane_detector.image_pipeline(image)
        
        plt.imsave("./project/result/frame%d.jpg" % idx, test_result)     # save frame as JPEG file

        #plt.imshow(test_result)
        #plt.show()

if __name__ == "__main__":
    main()

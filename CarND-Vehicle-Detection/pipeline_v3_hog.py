import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import time
import pickle
import collections
import os

from collections import deque
from itertools import chain

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def bin_spatial(img, size=(32, 32)):
    # Create the feature vector
    features = cv2.resize(img, size).ravel() 
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
#def color_hist(img, nbins=32, bins_range=(0, 1)):
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,demonstration=False):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # use all channels, which is identical to 
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    window_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            #print("hog features:", hog_features.shape)
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            #print("bin_spatial features:", spatial_features.shape)
            #print("hist features:", hist_features.shape)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if demonstration == True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                window_list.append(((xbox_left,  ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
            elif test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                 # Append window position to list
                window_list.append(((xbox_left,  ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def process_bboxes(image,hot_windows,threshold,show_heatmap=False):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)  
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,threshold)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #draw_img = draw_labeled_bboxes(np.copy(image), labels)
    bbox_list = find_labeled_boxes(labels)
    if show_heatmap:
        return bbox_list,heatmap
    else:
        return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def find_labeled_boxes(labels):
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
    return bbox_list

def draw_car_boxes(img, bbox_list):
    for bbox in bbox_list:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,255), 6)
    # Return the image
    return img

def pipeline(image):
    sample_image = image.copy()
    scales = [1,1.2,1.5]
    tmp_windows = []
    for scale in scales:
        hot_windows = find_cars(sample_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,demonstration=False)
        tmp_windows = hot_windows + tmp_windows
    #print("founded box number based SVC ..",len(tmp_windows))        

    # multi boxes drawn on raw image ...
    #result_window = draw_boxes(sample_image, tmp_windows, color=(0, 255, 255), thick=3)
    #result_windows.append(result_window)

    # draw heat map 
    bbox_list = process_bboxes(sample_image,tmp_windows,threshold=1,show_heatmap=False)
    draw_img = draw_car_boxes(sample_image, bbox_list)
    return draw_img


hot_windows_list = []
windows = []
i = 0
def process_image(image, n_frames=10, threshold=22):  
     
    hot_windows_temp =[]
    global hot_windows_list
    global windows
    global hot_windows_final
    global i


    sample_image = image.copy()
    scales = [1,1.2,1.5]
    tmp_windows = []
    for scale in scales:
        hot_windows = find_cars(sample_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,demonstration=False)
        tmp_windows = hot_windows + tmp_windows

    hot_windows_list.append(tmp_windows)
   
    if len(hot_windows_list) <= n_frames:
        hot_windows_final = sum(hot_windows_list, []) # Add windows from all available frames
        bbox_list = process_bboxes(sample_image,hot_windows_final,threshold=2,show_heatmap=False)

    #Look at last n frames and append all hot windows found
    else: 
        for val in hot_windows_list[(len(hot_windows_list) - n_frames -1) : (len(hot_windows_list)-1)]:
            hot_windows_temp.append(val)
        #Flatten this list
        hot_windows_final = sum(hot_windows_temp, [])
        bbox_list = process_bboxes(sample_image,hot_windows_final,threshold=22,show_heatmap=False)    
    
    
    #print("founded box number based SVC ..",len(tmp_windows))        
    #print("    final box numbers  ..",len(hot_windows_final)) 


    draw_img = draw_car_boxes(sample_image, bbox_list)
    
    font_size = 2
    text = 'FrameNo:' + str(i) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(draw_img,text,(10,650),font, font_size,(255,0,255))
    i += 1

    return draw_img


## defining a class to keep track of the previous bounding boxes
class VDetector:

    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    clf = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]    
    
    #scales = [1,1.2,1.5,1.8,2,2.4,3]
    scales = [1,1.5]

    ## initial list of variables to be passed in
    y_start_stop = [375, 640]
    color_space = 'YCrCb'
    hog_channel = True
    spatial_feat = True
    color_feat = True
    hog_feat = True
    
    ## initialize variables
    def __init__(self, frames_to_keep=20):
        self.hotbox_hist = deque(maxlen=frames_to_keep)
    
    def pipeline(self,image):
        sample_image = image.copy()
        
        tmp_windows = []
        for scale in self.scales:
            hot_windows = find_cars(sample_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,demonstration=False)
            tmp_windows = hot_windows + tmp_windows
        #print("founded box number based SVC ..",len(tmp_windows))        

        # multi boxes drawn on raw image ...
        #result_window = draw_boxes(sample_image, tmp_windows, color=(0, 255, 255), thick=3)
        #result_windows.append(result_window)

        # draw heat map 
        bbox_list = process_bboxes(sample_image,tmp_windows,threshold=1,show_heatmap=False)
        draw_img1 = draw_car_boxes(sample_image, bbox_list)
        #plt.imshow(draw_img1)
        #plt.show()

        ## process for previouse frame of video images
        self.hotbox_hist.append(bbox_list)
        #
        heatmap_mask = np.zeros_like(sample_image[:,:,0]).astype(np.float)  
        # Add heat to each box in box list
        heatmap_mask = add_heat(heatmap_mask, list(chain.from_iterable(self.hotbox_hist)))
        # Apply threshold to help remove false positives
        heatmap_mask = apply_threshold(heatmap_mask,threshold=1)
        # Visualize the heatmap when displaying    
        heat_img = np.clip(heatmap_mask, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heat_img)
        #draw_img = draw_labeled_bboxes(np.copy(image), labels)
        bbox_list = find_labeled_boxes(labels)        
        draw_img = draw_car_boxes(sample_image, bbox_list)
        

        return draw_img    

def main():


    frame_range = range(0,37)
    print("test image sizes", len(frame_range))

    project_images = [ os.path.join("./test", "frame" + str(r) + ".jpg") for r in frame_range ]
    imread_op = lambda x:mpimg.imread(x)
    p_images = list(map( imread_op, project_images )) 

    save_images = [ os.path.join("./test_save", "hog_found" + str(r) + ".jpg") for r in frame_range ]

    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    global svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins,ystart,ystop
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    y_start_stop = [375, 640]
    ystart, ystop = y_start_stop[0], y_start_stop[1]

    print("orient:%s" %  orient)
    print("pix_per_cell : %s" % pix_per_cell)
    print("cell_per_block : %s" % cell_per_block)
    print("spatial_size : " , spatial_size)
    print("hist_bins : %s" % hist_bins)
    print("y_start_stop : %s" % y_start_stop)

    print(" load X / y train test data ....")
    X_train =    np.load("X_train.npy"  )
    X_test =    np.load("X_test.npy"  )
    y_train =    np.load("y_train.npy"   )
    y_test =    np.load("y_test.npy"   )

    i = 0
    vd = VDetector()
    for p_savefile, sample_image in zip(save_images, p_images):
        draw_img = vd.pipeline(sample_image)
        plt.imsave(p_savefile,draw_img  )
        i += 1
    #video = VideoFileClip("project_video.mp4")
    #project_clip = video.fl_image(process_image) #NOTE: this function expects color images!!
    #output = "output_images/vehicle_detection.mp4"
    #project_clip.write_videofile(output, audio=False)

if __name__ == "__main__":
    main()
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_noncar]: ./result_images/car_noncar.jpeg
[hog_gray]: ./result_images/hog_gray.jpeg
[hog_car]: ./result_images/hog_ycrcb_car.jpeg
[hog_ncar]: ./result_images/hog_ycrcb_ncar.jpeg
[hog_car9]: ./result_images/hog_ycrcb9_car.jpeg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

# Histogram of Oriented Gradients (HOG)

## 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

`All program codes are put into one ipython (jupyter notebook.)`
***

## 1.1 Loading Training Data 

First of all, all training data pack is downloaded from following site.

[Vehicle]\
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip \
[Non Vehicle]\
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

I started reading in all the `vehicle` and `non-vehicle` images and then show images table on jupyter notebook, which displays **CAR** and **NON CAR** images side by side. Left side is **CAR** and Right side is **NON CAR** images.
Please take a look at below example respectively.
Note that in order to save display and image processing time, I have just picked up only 10 random images from CARS and NON CARS images.


![alt text][car_noncar]

In the next step to investigate how HOG image is working to extract normal image, I have setup following standard values for each main parameters. 

| Parameter name  | Setup Values    |
|:-------------:|:-------------:|
| orientations  | 9 |
| pix_per_cell     | 8  ,8  |
| cell_per_block    | 2, 2  |

As first experiment, I have extracted HOG features from GRAY image (`converted with cv2.cvtColor(im , cv2.COLOR_RGB2GRAY)`), which is identical instructions to text books. **(#20 scikit HOG image)** 


![alt text][hog_gray]

Another example using the `YCrCb` color space and use same HOG parameters used in HOG Gray image extraction.
Please take a look at the below images.

**HOG YCrCb Car**
![alt text][hog_car]
**HOG YCrCb Non Car**
![alt text][hog_ncar]


# 2. Explain how you settled on your final choice of HOG parameters.

As explained in previous section, I have decided to use **Y** channel of **YCrCb** and `orientations=9`, `pixel_per_cell=8`, `cell_per_block=2`. 
Since pixell_per_cell increased to `4`, we could get much precious feature points and densed bit image points on HOG image. As a result to change parameters, it is by no means cerfitication to get more accurate outcome for image identification. Thus, pixel_per_cell `2` would be best scenario for the time being upto we coordinate training process. 
However, those parameter combinations would be implemented so that classifier can reflect best score from training set of data.   

Here is sample table to show pixel_per_cell `4`.
![alt text][hog_car9]



# 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

**Car HOG**
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


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
[X_rgb]: ./result_images/X_rgb.jpeg
[X_y]: ./result_images/X_ycrcb.jpeg



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
Since pixell_per_cell increased to `4`, we could get much precious feature points and densed bit image points on HOG image. As a result to change parameters, it by no means gives credits to get more accurate outcome for image identification. Thus, pixel_per_cell `2` would be best scenario for the time being upto we coordinate training process. Als, its image is robust to find frame of the car and the non-car objects.
However, those parameter combinations would be implemented so that classifier can reflect best score from training set of data.   

Here is sample table to show pixel_per_cell `4`.
![alt text][hog_car9]



# 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

## 3.1 Color Feature Extraction
***

First of all, I have provide the training set, X and y, which consists of CAR/NON CAR features and labels for preparation of the training.
The features X is build 1) scaled down to 32x32 and make histogram summary (bin scaling = 32), then concatenate those array data to combined 
into single features.
I have tried to extract 2 different features patterns, one is from RGB image, another one is from YCrCb conversion image data.
Following is sample data show.

**RGB** \
![alt text][X_rgb]
**YCrCb** \
![alt text][X_y]

1st version Feature extraction algorythm is exerted from text book as following.
I choosed "RGB" for raw_image setting. (image is converteed to none of other types.)

```
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        # Read in each one by one
        #image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features
        
```

After building up X, y, I have applied svc training process to gain best score performance of the model.
Initially, I obtained 95% accuracy score from SVC binary classification.

```
## Train / Test Data split 
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# standarized (normalized data)

X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to both X_train and X_test
scaled_X_train = X_scaler.transform(X_train)
scaled_X_test = X_scaler.transform(X_test)

from sklearn.svm import LinearSVC
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(scaled_X_train, y_train)

print('Test Accuracy of SVC = ', svc.score(scaled_X_test, y_test))

Test Accuracy of SVC =  0.9504504504504504
```

## 3.2 HOG Feature Extraction

Now I have added one additional feature layer extracting HOG features from raw image.
So, final features style is 
binScale + histogram + HOG features = total 4932 byte features
As a result of running Support Vector algo with the defaut parameters, I was able to show best accuracy score performance 
from those features data set. I got 98% accuracy !!


```
eprecated and will be changed to `L2-Hys` in v0.15
  'be changed to `L2-Hys` in v0.15', skimage_deprecation)
 loaded car features --> 8792 (4932,)
 loaded non car features --> 8968 (4932,)
  Buildng X y training data set .....
 Spliting data set to train == 80% / test == 20%
 Scaling (Normalized data) ....
Suuport Vector training starting ....
 Check scoring ....
Test Accuracy of SVC =  0.9817004504504504
```

## 3.3 YCrCb
YCrCb is more robust image data set to distinguish color scheme rather than RGB. When I applied YCrCb converted image and build HOG features extraction for all channels, `the accuracy score is dramatically increased up to 99%` !! (features byte is extended to 8640 bytes)

```
(carnd-term1) milano:CarND-Vehicle-Detection donchan$ python svc_train.py --color="YCrCb" --hog_channel=3
------------------------------
* Color Scheme ... * YCrCb
* HOG channel ... * 3
------------------------------
Number of Cars Data .. 8792
Number of Non Cars Data .. 8968
/Users/donchan/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/skimage/feature/_hog.py:119: skimage_deprecation: Default value of `block_norm`==`L1` is deprecated and will be changed to `L2-Hys` in v0.15
  'be changed to `L2-Hys` in v0.15', skimage_deprecation)
 loaded car features --> 8792 (8460,)
 loaded non car features --> 8968 (8460,)
  Buildng X y training data set .....
 Spliting data set to train == 80% / test == 20%
 Scaling (Normalized data) ....
Suuport Vector training starting ....
 Check scoring ....
Test Accuracy of SVC =  0.9912725225225225
```


# Sliding Window Search

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


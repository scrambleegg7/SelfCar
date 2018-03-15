# My Writeup Report


---

**Advanced Lane Finding Project**


It is 4th assigned project of UdaCity Nanodegree Self-Driving Car Course. It needs deep image processing to detect straight line 
and curved line after image conversion to binary masked image. 

The goals / steps of this project we have to address are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[drawchess]: ./results_images/drawchess.jpeg "Grayscale"
[undistortchess]: ./results_images/undistortchess.jpeg "undistort"
[missingchess]: ./results_images/missingchess.jpeg "missing"
[sobelx]: ./results_images/sobelx.jpeg "sobelx"
[sobely]: ./results_images/sobely.jpeg "sobely"
[rgb]: ./results_images/rgb.jpeg "rgb"
[normal_undistort]: ./results_images/normal_undistort.jpeg "n"
[mixed]: ./results_images/mixed.jpeg "ixed"
[mag]: ./results_images/mag.jpeg "magnitude"
[dir]: ./results_images/dir.jpeg "direction"
[sobelcombine]: ./results_images/sobelcombine.jpeg "direction"
[hls_color]: ./results_images/hls_color.jpeg "rgb"
[hls_binary]: ./results_images/hls_binary.jpeg "rgb"

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

---

# Camera Calibration (Part1)

## 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This part is described and program coded on **calibration.ipynb** jupyter notebook under root directory "CarND-Advanced-Lane-Lines. 


First of all, I started by preparing to build python class module ChessBoardClass, which encupsuled some of siginificant methods for setting full set of test images, and checking chessboard corners.
Following process are cordinatd through this class module method.

__*Initial Process of class module*__:
class module has initial parameters to set the default chess size. By visual confirmation, I have setup 9 for x-axis and 6 for y-axis for number of crosspoints of chessboards. Those parameters are used for checkChessBoard method. 

1. **setImage method**: Image filenames are set and then open skimage imread function. Then I converted 3 channel colored image to gray image, which of the dimention is downgraded to 1. I have used standard cv2.cvtCOLOR function with parameter COLOR_RGB2GRAY. 

2. **checkCessBoard** method: 
After converting gray image from original colored image, all of incoming images are saved in the memory (gray_images list) of class module, then corner points are searched with cv2.findChessboard function by iterating gray_images list. If success to find the corners of target chessboard, obj points and corners list returned with the function are saved in objpoints and imgpoints list respectively. Finally cv2.drawChessboardCorners function draw the corner points on the original images. Please see the result image from drawChessBoaardCorners as following.
Also, the methods save unidentified chessboards, on which cv2 function can hardly detect the corners.   

3. **undistort** method:
I have applied cv2.calibrateCamera so that it simply outputs several siginificant parameters (mts / dist) from saved objpoints and imgpoints list, those 2 key ones are used for cv2.undistort function to generate undistortion image. After applying cv2.undistort against original image, I have saved ones and show the correct chessboards  on the attached source. They have rather straight lines than original ones on each top and bottom of images.   

4. **saveCalibParameters**
I have kept mtx and dist parameters from cv2.calibrateCamera function. Those 2 important parameters are passed to further undistortion function of reading other standard photo images. 
Therefore, I have saved **mtx** and **dist** parameters on pickle file. 
The filename __calibration.p__ has been saved under ./picked_data directory.  

### It is successful to find the corners of the chessboards and then draw the line and corner points.
![alt text][drawchess]

### Those are missing images by findChessboardCorners.
![alt text][missingchess]

### Those are undistortion images by cv2.undistort, after cameraClibration function using saved objpoints and imgpoints.
![alt text][undistortchess]

---

# Pipeline Image process for binary masking (Part2)

 This is written on `pipelineTestImage.ipynb` juyter notebook for the demonstration how I build transfered (threshhold) images from original road images.

## 1. Provide an example of a distortion-corrected image.

With using cv2.undistort function and saved parameters **mtx** and **dist**, I have successfully transformed the undistortion image from original road images. It is confirmed that images are adjusted based on the camera calibration parameters, in special distortion images (eg. slightly curved lines) are clearly purged from original ones. For example, the back of white car is truncated from the right side of image window. 
I have demonstrated original and adjusted images side by side on jupyter notebook.


![alt text][normal_undistort]

## 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

### 2.1 Split Channel RGB colored image
At first step, I splitted out Red, Green, Blue channel from original one to find out which channel has strong features for the lane line. I have confirmed that Red and Green channel showed us good image contrasts to draw the line on separated Gray image. Blue was unfortunately poor result.

![alt text][rgb]


### 2.2 Extract Yellow & White & HLS Yellow to make mixed mask image

I have set several color parameters to extract yellow and white color from original images. (`threshColoredImageBin submodule of utils.py`)
Target colored lines are drawn accordingly on each binary image and then finally those are put together into one single binary image by switching on True condition of each filtering image.
Overall, we have found that HLS Yellow line is better displayed from the bottom of the image window, which of length is much longer than simple Yellow line binary extraction.  
Finally, I have combined all of three binary images so that I would show the effectiveness of consolidated binary image.
It is apparently 2 lane lines are drawn on the integrated binary image.   

```
    # rgb Yellow colored mask
    lower = np.array([255,180,0]).astype(np.uint8)
    upper = np.array([255,255,170]).astype(np.uint8)

    # rgb white colored mask
    lower = np.array([100,100,200]).astype(np.uint8)
    upper = np.array([255,255,255]).astype(np.uint8)

    # HLS yellow fitltering
    lower = np.array([20,120,80]).astype(np.uint8)
    upper = np.array([45,200,255]).astype(np.uint8)
    
```
![alt text][mixed]

### 2.3 Sobel X / Y Direction
At 2nd step, I have applied Sobel Gradient technique against original images (undistortion). Following tables indicates what kinds of parameters I have tested.

 | Parameter        | value   | 
|:-------------:|:-------------:| 
| direction     | x  & y      | 
| thresh_min      | 20       | 
| thresh_max     | 100    | 
| kernel size     | 3, 5, 15       | 

After obtaining result from sobel gradient techiques, image having kernel size = 15 has remarkable contrast that we could easily identify thick lane line on the binary transformed images.
I tried to show 2 different direction, X and Y from original images, but there are some noisy tones dislayed on the generated images of Sobel Y.

#### solbelX
![alt text][sobelx]
#### sobelY
![alt text][sobely]



### 2.4 Magnitude Sobel
Next I have applied Magnitude Sobel Technique, which has combination of Sobel X and Y direction Gradient image. Though we see kernel size = 15 is best performance to show the contrast, some lane lines are fully depicted on both test4 and test5 images. Because those 2 images have big color contrast on road surface, therefore it is hard to catch lane lines as continuous ones. 

 | Parameter        | value   | 
|:-------------:|:-------------:| 
| direction     | x and y       | 
| thresh_min      | 20       | 
| thresh_max     | 150    | 
| kernel size     | 3, 7, 15       | 

![alt text][mag]


### 2.5 Direction of Gradient 
Finally I have tested Direction of Gradient Technique as single testing module. Following are 2 main parameters I have setup for testing purposes to see effect of filtering with different parameters. The `thresh-2`, a.k.a (0.7,1.3) is same setting as Text book suggested to test image. Clearly, we found thresh-2 parameter gave good contrast performance of filtering images, where we see lane lines are deciphered from snow-style background image.  


| Parameter        | value   | 
|:-------------:|:-------------:| 
| thresh-1       |    (0, np.pi / 2)  | 
| thresh-2    | (0.7, 1.3)    | 

![alt text][dir]


### 2.6 Combination of Sobel / Magnitude / Direction of Gradient  
Then I have integrated above 4 techniques to make blend image showing lane line with the binary image as following code. I have picked up best parameters for respective gradient techniques. I obtained good result overall from mixed images, however some of images like test4 and test5 has lost contrast on upper half of image size, in a result to have incapability of drawing full length of lane lines.

```
def applyCombinedGradient(x):
    
    sobel_imagex = abs_sobel_thresh(x, orient='x', thresh_min=20, thresh_max=120, kernel_size=15)
    sobel_imagey = abs_sobel_thresh(x, orient='y', thresh_min=20, thresh_max=120, kernel_size=15)

    mag_binary = mag_thresh(x, sobel_kernel=15, mag_thresh=(80, 200))
    dir_binary = dir_threshold(x, sobel_kernel=15, thresh=(np.pi/4, np.pi/2) )
    
    mybinary = np.zeros_like(dir_binary)
    mybinary[ ((sobel_imagex == 1) & (sobel_imagey == 1)) | ( (mag_binary == 1) & (dir_binary == 1)      )      ] = 1
    #mybinary[(sobel_imagex == 1) | ((sobel_imagey == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1
    return mybinary
```
![alt text][sobelcombine]


In order to grab more contrast for Yellow and White line on the road, I also applied HSL binary image filtering.
One is to use full channels of HLS extracting Yellow and White line, another is to focus on just S channel having threshold parameters for Yellow and White line.

**My final conclusion which is best criteria highlighting Yellow and White lane line:**
S channel + Sobel Gradient Filtering is best combination to draw the full length of lane line on the road binary image. 

HLS Yellow 
```
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # HLS yellow colored mask
    lower = np.array([20,120,80]).astype(np.uint8)
    upper = np.array([45,200,255]).astype(np.uint8)
    mask = cv2.inRange(hls,lower,upper)
    hls_y = cv2.bitwise_and(image, image, mask=mask)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)    
    gray = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    y_hls_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

    # white colored mask
    lower = np.array([0,200,0]).astype(np.uint8)
    upper = np.array([255,255,255]).astype(np.uint8)
    mask = cv2.inRange(hls,lower,upper)
    hls_w = cv2.bitwise_and(image, image, mask=mask)
    hls_w = cv2.cvtColor(hls_w, cv2.COLOR_HLS2RGB)    
    gray = cv2.cvtColor(hls_w, cv2.COLOR_RGB2GRAY)
    w_binary = threshold(gray, bin_thresh_min, bin_thresh_max)

```

```
    # input is RGB (skimage.io.imread)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]  # 2 <-- S channel     
    
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

```

Then regenerated 2types of images - Colored images (3 channels image, but channel0 is ZERO pad, and channel1 is sobel gradient combined image, and channel2 is HLS filtering image.) and binary images to extract sobel gradient combined and HLS filtering.

#### Colored Image

![alt text][hls_color]

#### Integrated Binary Image

![alt text][hls_binary]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is written on `undistort_corners_warp` module. After grabing gray conversion image shape, original trapezoid shape defined as **src** array
has been transfered to squre style polygon, which region is defined as **dst**.  **src** and **dst** are hardcoded parameters on program code.
Then, Opencv perspectivetransform and warpPerspective are used to make bird view images.  

```python
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (gray.shape[1], gray.shape[0])

    src = np.float32([[180, img_size[1]], [575, 460], 
                    [705, 460], [1150, img_size[1]]])

    dst = np.float32([[250, img.shape[0]], [250, 0], 
                      [960, 0], [960, img.shape[0]]])


```

Following source and destination points are used in `undistort_corners_warp`

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 1250      | 250, 720        | 
| 575, 460      | 250, 0      |
| 705, 460     | 960, 0      |
| 1150,       | 960, 720        |


For the visual verification, I have generated comaprison image matrix for normal and warped images as follows;
Straight lane lines and curved lane lines are confirmed to draw the paralell ones on respective images. 

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

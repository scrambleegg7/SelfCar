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
[undistortchess]: ./results_images/undistortchess.jpeg "Grayscale"
[missingchess]: ./results_images/missingchess.jpeg "Grayscale"

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
Mainly, those are used for my part2 section.  

### It is successful to find the corners of the chessboards and then draw the line and corner points.
![alt text][drawchess]

### Those are missing images by findChessboardCorners.
![alt text][missingchess]

### Those are undistortion images by cv2.undistort, after cameraClibration function using saved objpoints and imgpoints.
![alt text][undistortchess]

# Pipeline Image process for binary masking (Part2)

 


For the further usage other program codes will access, I have saved **mtx** and **dist** parameters on pickle file. 
The filename __calibration.p__ has been saved under ./picked_data directory.  


# Pipeline (single images)

## 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

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

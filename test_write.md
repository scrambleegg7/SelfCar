
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

import imageio
imageio.plugins.ffmpeg.download()


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML, display
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7fa6b04f3f28>




![png](output_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drawLine(img, x, y, color=[255, 0, 0], thickness=20):
    """
    Adjust a line to the points [`x`, `y`] and draws it on the image `img` using `color` and `thickness` for the line.
    """
    if len(x) == 0: 
        return
    
    lineParameters = np.polyfit(x, y, 1) 
    
    m = lineParameters[0]
    b = lineParameters[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b)/m)
    y2 = int((maxY/2)) + 60
    x2 = int((y2 - b)/m)
    cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 4)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            
            # slope < 0 --> right lane
            # slope > 0 --> left lane
            slope= (y1 - y2)/(x1 - x2)
            if slope < 0:
                #print("left slope -> x1:%d y1:%d x2:%d y2:%d" % (x1,y1,x2,y2)  )
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
                #pass
            else:
                #print("right slope -> x1:%d y1:%d x2:%d y2:%d" % (x1,y1,x2,y2)  )
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)
                #pass
            
    drawLine(img, leftPointsX,  leftPointsY,  color, thickness)
    drawLine(img, rightPointsX, rightPointsY, color, thickness)
    #drawLine(img, x, y, color=[255, 0, 0], thickness=20)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    #
    # call draw lines ....
    # split line dot to integrate into single lines 
    #
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```

## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os

testdir = "test_images"
testimagenames = os.listdir("test_images/")
```


```python
def showImagesInHtml(images, dir):
    """
    Shows the list of `images` names on the directory `dir` as HTML embeded on the page.
    """
    randomNumber = 1
    buffer = "<div>"
    for img in images:
        imgSource = dir + '/' + img + "?" + str(randomNumber)
        print(imgSource)
        buffer += """<img src="{0}" width="300" height="110" style="float:left; margin:1px"/>""".format(imgSource)
        randomNumber += 1
    buffer += "</div>"
    display(HTML(buffer))

```


```python
def saveImages(images, outputDir, imageNames, isGray=0):
    """
    Writes the `images` to the `outputDir` directory using the `imagesNames`.
    It creates the output directory if it doesn't exists.
    
    Example:
    saveImages([img1], 'tempDir', ['myImage.jpg'])
    Will save the image on the path: tempDir/myImage.jpg
    
    """
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    zipped = list(map(lambda imgZip: (outputDir + '/' + imgZip[1], imgZip[0]), zip(images, imageNames)))
    for imgPair in zipped:
        if isGray:
            plt.imsave(imgPair[0], imgPair[1], cmap='gray')
        else :
            plt.imsave(imgPair[0], imgPair[1])
```


```python
def maskEdge(img):
    # select area map from main image to fit lane line
    # in maskEdge routine, human(programmer) intentionaly need to define area where lane is drawn on the road.
    # otherwise, system has no ability to find where lane is detected.
    # It should have further visual process development in future enhancements to detect lane line processing 
    # with automatic.
    ysize = img.shape[0]
    xsize = img.shape[1]
    region = np.array([ [0, ysize], [xsize/2,(ysize/2)+ 10], [xsize,ysize] ], np.int32)
    
    #
    # create region for trapezoid rather than defining triangle area.
    # in order to catch up line more preciousely
    #
    #region = np.array([ [0, ysize], [.45 * xsize, .62 * ysize], [.55 * xsize, .62 * ysize], [xsize ,ysize] ], np.int32)
    
    return region_of_interest(img, [region])
```


```python
def myPipeline(in_image):
    #
    # define parameter for gaussian
    #
    kernel_size = 15

    #
    # define paraeters for canny process 
    #
    low_threshold = 20
    high_threshold = 100

    rho = 1 #3
    theta = np.pi/180
    threshold = 10
    min_line_length = 20
    max_line_gap = 1
    
    grayAction = lambda img : grayscale(img)
    gaussianAction = lambda img : gaussian_blur(img, kernel_size)
    cannyAction = lambda img : canny(img, low_threshold, high_threshold)
    maskAction = lambda img : maskEdge(img) 
    houghAction = lambda img: hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
    
    line_image =  houghAction( maskAction( cannyAction( gaussianAction( grayAction(in_image) ) ) ) )
    
    weightedAction = lambda imgs: weighted_img(imgs[0], imgs[1])
    
    result = weightedAction( [ line_image, in_image ]     )
    
    return result # combined image (line + original)
```

# Show Original TestImage(s) #


```python
# show images with HTML format....

showImagesInHtml(testimagenames,testdir)
```

    test_images/solidYellowCurve.jpg?1
    test_images/solidYellowLeft.jpg?2
    test_images/solidYellowCurve2.jpg?3
    test_images/solidWhiteRight.jpg?4
    test_images/whiteCarLaneSwitch.jpg?5
    test_images/solidWhiteCurve.jpg?6



<div><img src="test_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>



```python
testImagesList = list(map(lambda img: plt.imread(testdir + '/' + img), testimagenames))
```


```python
# total test images are put into one list...

print("** Total test image number in testImagesList --> %d" %   len(testImagesList) )
```

    ** Total test image number in testImagesList --> 6


# Convert Gray Scale (Preparation) #


```python
grayImagesList = list(map( lambda img : grayscale(img)  , testImagesList  ) )
```


```python
outputDir = "test_gray_images"
saveImages(grayImagesList, outputDir, testimagenames, isGray=1)
showImagesInHtml(testimagenames,outputDir)
```

    test_gray_images/solidYellowCurve.jpg?1
    test_gray_images/solidYellowLeft.jpg?2
    test_gray_images/solidYellowCurve2.jpg?3
    test_gray_images/solidWhiteRight.jpg?4
    test_gray_images/whiteCarLaneSwitch.jpg?5
    test_gray_images/solidWhiteCurve.jpg?6



<div><img src="test_gray_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_gray_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_gray_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_gray_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_gray_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_gray_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


# Apply Gaussian Blur #


```python
kernel_size = 5
gaussImagesList = list(map( lambda img : gaussian_blur(img, kernel_size)  , grayImagesList  ) )
```


```python
gauss_outputDir = "test_gauss_images"
saveImages(gaussImagesList, gauss_outputDir, testimagenames, isGray=1)
showImagesInHtml(testimagenames,gauss_outputDir)
```

    test_gauss_images/solidYellowCurve.jpg?1
    test_gauss_images/solidYellowLeft.jpg?2
    test_gauss_images/solidYellowCurve2.jpg?3
    test_gauss_images/solidWhiteRight.jpg?4
    test_gauss_images/whiteCarLaneSwitch.jpg?5
    test_gauss_images/solidWhiteCurve.jpg?6



<div><img src="test_gauss_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_gauss_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_gauss_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_gauss_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_gauss_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_gauss_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


# Apply Canny Action against Gaussian Blur images #


```python
low_threshold = 50
high_threshold = 150
cannyImagesList = list(  map( lambda img : canny(img, low_threshold, high_threshold)  , gaussImagesList  )   )
```


```python
#
# save canny images into test_canny_images directory 
#
canny_outputDir = "test_canny_images"
saveImages(cannyImagesList, canny_outputDir, testimagenames, isGray=1)
# show image on html format
showImagesInHtml(testimagenames,canny_outputDir)
```

    test_canny_images/solidYellowCurve.jpg?1
    test_canny_images/solidYellowLeft.jpg?2
    test_canny_images/solidYellowCurve2.jpg?3
    test_canny_images/solidWhiteRight.jpg?4
    test_canny_images/whiteCarLaneSwitch.jpg?5
    test_canny_images/solidWhiteCurve.jpg?6



<div><img src="test_canny_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_canny_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_canny_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_canny_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_canny_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_canny_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


# Hough Images #

# Create Masked Image #


```python
maskImagesList = list(  map( lambda img : maskEdge(img)  , cannyImagesList  )   )
```


```python
#
# save masked images into test_masked_images directory 
#
masked_outputDir = "test_masked_images"
saveImages(maskImagesList, masked_outputDir, testimagenames, isGray=1)
# show image on html format
showImagesInHtml(testimagenames,masked_outputDir)
```

    test_masked_images/solidYellowCurve.jpg?1
    test_masked_images/solidYellowLeft.jpg?2
    test_masked_images/solidYellowCurve2.jpg?3
    test_masked_images/solidWhiteRight.jpg?4
    test_masked_images/whiteCarLaneSwitch.jpg?5
    test_masked_images/solidWhiteCurve.jpg?6



<div><img src="test_masked_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_masked_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_masked_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_masked_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_masked_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_masked_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


## read masked images from canny images to omit unnecesary area except lane lines ##


```python
#
#    Hough Definition to detect slope line 
#

rho = 1 #3
theta = np.pi/180
#threshold = 20
threshold = 10
min_line_length = 30
#min_line_length = 40
max_line_gap = 1
#max_line_gap = 20
#line_image = np.copy(image)*0 #creating a blank to draw lines on

houghAction = lambda img: hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
houghLineImagesList = list(  map( houghAction, maskImagesList  )   )
```


```python
#
# save hough images into test_canny_images directory 
#
hough_outputDir = "test_hough_images"
saveImages(houghLineImagesList, hough_outputDir, testimagenames, isGray=1)
# show image on html format
showImagesInHtml(testimagenames, hough_outputDir)
```

    test_hough_images/solidYellowCurve.jpg?1
    test_hough_images/solidYellowLeft.jpg?2
    test_hough_images/solidYellowCurve2.jpg?3
    test_hough_images/solidWhiteRight.jpg?4
    test_hough_images/whiteCarLaneSwitch.jpg?5
    test_hough_images/solidWhiteCurve.jpg?6



<div><img src="test_hough_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_hough_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_hough_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_hough_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_hough_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_hough_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


# Weighted Images (combine original images with Lined Image #


```python
weightedAction = lambda imgs: weighted_img(imgs[0], imgs[1])
weightedImagesList = list(  map( weightedAction, zip( houghLineImagesList, testImagesList ) )   )
```


```python
#
# save weighted combined images into test_combine_images directory 
#
combine_outputDir = "test_combine_images"
saveImages(weightedImagesList, combine_outputDir, testimagenames, isGray=1)
# show image on html format
showImagesInHtml(testimagenames, combine_outputDir)
```

    test_combine_images/solidYellowCurve.jpg?1
    test_combine_images/solidYellowLeft.jpg?2
    test_combine_images/solidYellowCurve2.jpg?3
    test_combine_images/solidWhiteRight.jpg?4
    test_combine_images/whiteCarLaneSwitch.jpg?5
    test_combine_images/solidWhiteCurve.jpg?6



<div><img src="test_combine_images/solidYellowCurve.jpg?1" width="300" height="110" style="float:left; margin:1px"/><img src="test_combine_images/solidYellowLeft.jpg?2" width="300" height="110" style="float:left; margin:1px"/><img src="test_combine_images/solidYellowCurve2.jpg?3" width="300" height="110" style="float:left; margin:1px"/><img src="test_combine_images/solidWhiteRight.jpg?4" width="300" height="110" style="float:left; margin:1px"/><img src="test_combine_images/whiteCarLaneSwitch.jpg?5" width="300" height="110" style="float:left; margin:1px"/><img src="test_combine_images/solidWhiteCurve.jpg?6" width="300" height="110" style="float:left; margin:1px"/></div>


## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
```

## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = myPipeline(image)
    return result
```


```python
def videoProcessing(videoFilename, videoInDirectory, videOutDirectory):

    if not os.path.exists(videOutDirectory):
        os.makedirs(videOutDirectory)   
    
    inVideoFile = videoInDirectory + "/" + videoFilename
    outVideoFile = videOutDirectory + "/" + videoFilename
    
    clip1 = VideoFileClip(inVideoFile)
    white_clip = clip1.fl_image(process_image)
    %time white_clip.write_videofile(outVideoFile, audio=False)
    
    return outVideoFile
```

Let's try the one with the solid white lane on the right first ...


```python
videoFilenameW = "solidWhiteRight.mp4"
videoFilenameY = "solidYellowLeft.mp4"

videoInDirectory = "test_videos"
videOutDirectory = "test_videos_output"

white_output = videoProcessing(videoFilenameW, videoInDirectory, videOutDirectory)

#white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    
    
      0%|          | 0/222 [00:00<?, ?it/s][A[A
    
      4%|▎         | 8/222 [00:00<00:02, 73.14it/s][A[A
    
      8%|▊         | 17/222 [00:00<00:02, 75.56it/s][A[A
    
     12%|█▏        | 26/222 [00:00<00:02, 78.79it/s][A[A
    
     16%|█▌        | 35/222 [00:00<00:02, 80.33it/s][A[A
    
     20%|█▉        | 44/222 [00:00<00:02, 82.01it/s][A[A
    
     23%|██▎       | 52/222 [00:00<00:02, 76.97it/s][A[A
    
     27%|██▋       | 60/222 [00:00<00:02, 75.17it/s][A[A
    
     31%|███       | 68/222 [00:00<00:02, 73.64it/s][A[A
    
     34%|███▍      | 75/222 [00:00<00:02, 70.00it/s][A[A
    
     37%|███▋      | 83/222 [00:01<00:01, 71.29it/s][A[A
    
     41%|████      | 90/222 [00:01<00:01, 70.23it/s][A[A
    
     44%|████▍     | 98/222 [00:01<00:01, 70.53it/s][A[A
    
     47%|████▋     | 105/222 [00:01<00:01, 69.97it/s][A[A
    
     50%|█████     | 112/222 [00:01<00:01, 69.36it/s][A[A
    
     54%|█████▎    | 119/222 [00:01<00:01, 68.52it/s][A[A
    
     57%|█████▋    | 126/222 [00:01<00:01, 68.91it/s][A[A
    
     60%|█████▉    | 133/222 [00:01<00:01, 68.23it/s][A[A
    
     63%|██████▎   | 140/222 [00:01<00:01, 67.53it/s][A[A
    
     66%|██████▌   | 147/222 [00:02<00:01, 66.28it/s][A[A
    
     69%|██████▉   | 154/222 [00:02<00:01, 65.92it/s][A[A
    
     73%|███████▎  | 161/222 [00:02<00:00, 67.04it/s][A[A
    
     76%|███████▌  | 168/222 [00:02<00:00, 66.82it/s][A[A
    
     79%|███████▉  | 176/222 [00:02<00:00, 69.02it/s][A[A
    
     82%|████████▏ | 183/222 [00:02<00:00, 67.53it/s][A[A
    
     86%|████████▌ | 190/222 [00:03<00:01, 27.71it/s][A[A
    
     89%|████████▊ | 197/222 [00:03<00:00, 33.34it/s][A[A
    
     92%|█████████▏| 205/222 [00:03<00:00, 39.89it/s][A[A
    
     96%|█████████▌| 213/222 [00:03<00:00, 46.18it/s][A[A
    
    100%|█████████▉| 221/222 [00:03<00:00, 52.29it/s][A[A
    
    [A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4 
    
    CPU times: user 9.84 s, sys: 216 ms, total: 10.1 s
    Wall time: 3.91 s



```python
yellow_output = videoProcessing(videoFilenameY, videoInDirectory, videOutDirectory)

```

    [MoviePy] >>>> Building video test_videos_output/solidYellowLeft.mp4
    [MoviePy] Writing video test_videos_output/solidYellowLeft.mp4


    
    
      0%|          | 0/682 [00:00<?, ?it/s][A[A
    
      1%|          | 8/682 [00:00<00:08, 76.82it/s][A[A
    
      2%|▏         | 17/682 [00:00<00:08, 77.77it/s][A[A
    
      4%|▍         | 26/682 [00:00<00:08, 79.87it/s][A[A
    
      5%|▌         | 36/682 [00:00<00:07, 82.70it/s][A[A
    
      7%|▋         | 45/682 [00:00<00:07, 81.71it/s][A[A
    
      8%|▊         | 52/682 [00:00<00:08, 75.07it/s][A[A
    
      9%|▊         | 59/682 [00:00<00:08, 72.50it/s][A[A
    
     10%|▉         | 66/682 [00:00<00:08, 71.73it/s][A[A
    
     11%|█         | 74/682 [00:00<00:08, 71.75it/s][A[A
    
     12%|█▏        | 81/682 [00:01<00:08, 70.40it/s][A[A
    
     13%|█▎        | 88/682 [00:01<00:08, 70.05it/s][A[A
    
     14%|█▍        | 96/682 [00:01<00:08, 71.17it/s][A[A
    
     15%|█▌        | 104/682 [00:01<00:08, 71.08it/s][A[A
    
     16%|█▋        | 112/682 [00:01<00:08, 71.10it/s][A[A
    
     18%|█▊        | 120/682 [00:01<00:07, 71.25it/s][A[A
    
     19%|█▉        | 128/682 [00:01<00:07, 70.30it/s][A[A
    
     20%|█▉        | 136/682 [00:01<00:07, 68.88it/s][A[A
    
     21%|██        | 143/682 [00:01<00:07, 68.57it/s][A[A
    
     22%|██▏       | 150/682 [00:02<00:07, 67.54it/s][A[A
    
     23%|██▎       | 157/682 [00:02<00:07, 66.89it/s][A[A
    
     24%|██▍       | 164/682 [00:02<00:07, 67.33it/s][A[A
    
     25%|██▌       | 171/682 [00:02<00:07, 67.10it/s][A[A
    
     26%|██▌       | 179/682 [00:02<00:07, 68.88it/s][A[A
    
     27%|██▋       | 186/682 [00:02<00:08, 60.68it/s][A[A
    
     28%|██▊       | 194/682 [00:02<00:07, 63.28it/s][A[A
    
     29%|██▉       | 201/682 [00:02<00:07, 64.21it/s][A[A
    
     30%|███       | 208/682 [00:02<00:07, 65.17it/s][A[A
    
     32%|███▏      | 215/682 [00:03<00:07, 65.72it/s][A[A
    
     33%|███▎      | 222/682 [00:03<00:06, 66.64it/s][A[A
    
     34%|███▎      | 230/682 [00:03<00:06, 68.02it/s][A[A
    
     35%|███▍      | 237/682 [00:03<00:06, 67.74it/s][A[A
    
     36%|███▌      | 244/682 [00:03<00:06, 67.24it/s][A[A
    
     37%|███▋      | 251/682 [00:03<00:06, 66.85it/s][A[A
    
     38%|███▊      | 259/682 [00:03<00:06, 69.18it/s][A[A
    
     39%|███▉      | 266/682 [00:03<00:06, 67.03it/s][A[A
    
     40%|████      | 273/682 [00:03<00:06, 67.65it/s][A[A
    
     41%|████      | 280/682 [00:04<00:05, 67.07it/s][A[A
    
     42%|████▏     | 288/682 [00:04<00:05, 68.53it/s][A[A
    
     43%|████▎     | 296/682 [00:04<00:05, 69.35it/s][A[A
    
     44%|████▍     | 303/682 [00:04<00:05, 69.43it/s][A[A
    
     45%|████▌     | 310/682 [00:04<00:05, 68.54it/s][A[A
    
     46%|████▋     | 317/682 [00:04<00:05, 66.10it/s][A[A
    
     48%|████▊     | 324/682 [00:04<00:05, 66.32it/s][A[A
    
     49%|████▊     | 331/682 [00:04<00:05, 67.12it/s][A[A
    
     50%|████▉     | 339/682 [00:04<00:04, 69.77it/s][A[A
    
     51%|█████     | 347/682 [00:05<00:05, 66.60it/s][A[A
    
     52%|█████▏    | 354/682 [00:05<00:04, 66.31it/s][A[A
    
     53%|█████▎    | 362/682 [00:05<00:04, 67.96it/s][A[A
    
     54%|█████▍    | 369/682 [00:05<00:04, 67.57it/s][A[A
    
     55%|█████▌    | 377/682 [00:05<00:04, 68.49it/s][A[A
    
     56%|█████▋    | 384/682 [00:05<00:04, 66.78it/s][A[A
    
     57%|█████▋    | 391/682 [00:05<00:04, 67.19it/s][A[A
    
     59%|█████▊    | 399/682 [00:05<00:04, 68.43it/s][A[A
    
     60%|█████▉    | 406/682 [00:05<00:04, 67.21it/s][A[A
    
     61%|██████    | 413/682 [00:05<00:04, 66.84it/s][A[A
    
     62%|██████▏   | 420/682 [00:06<00:03, 66.53it/s][A[A
    
     63%|██████▎   | 427/682 [00:06<00:03, 67.29it/s][A[A
    
     64%|██████▍   | 435/682 [00:06<00:03, 69.36it/s][A[A
    
     65%|██████▍   | 442/682 [00:06<00:03, 68.64it/s][A[A
    
     66%|██████▌   | 449/682 [00:06<00:03, 68.67it/s][A[A
    
     67%|██████▋   | 457/682 [00:06<00:03, 71.25it/s][A[A
    
     68%|██████▊   | 465/682 [00:06<00:03, 69.33it/s][A[A
    
     69%|██████▉   | 472/682 [00:06<00:03, 68.97it/s][A[A
    
     70%|███████   | 480/682 [00:06<00:02, 71.48it/s][A[A
    
     72%|███████▏  | 488/682 [00:07<00:02, 70.65it/s][A[A
    
     73%|███████▎  | 496/682 [00:07<00:02, 71.08it/s][A[A
    
     74%|███████▍  | 504/682 [00:07<00:02, 70.15it/s][A[A
    
     75%|███████▌  | 512/682 [00:07<00:02, 69.33it/s][A[A
    
     76%|███████▌  | 519/682 [00:07<00:02, 67.73it/s][A[A
    
     77%|███████▋  | 526/682 [00:07<00:02, 67.25it/s][A[A
    
     78%|███████▊  | 533/682 [00:07<00:02, 67.45it/s][A[A
    
     79%|███████▉  | 540/682 [00:07<00:02, 67.10it/s][A[A
    
     80%|████████  | 547/682 [00:07<00:01, 67.65it/s][A[A
    
     81%|████████  | 554/682 [00:08<00:01, 67.24it/s][A[A
    
     82%|████████▏ | 562/682 [00:08<00:01, 69.58it/s][A[A
    
     83%|████████▎ | 569/682 [00:08<00:01, 67.85it/s][A[A
    
     85%|████████▍ | 577/682 [00:08<00:01, 68.74it/s][A[A
    
     86%|████████▌ | 584/682 [00:08<00:01, 68.51it/s][A[A
    
     87%|████████▋ | 591/682 [00:08<00:01, 66.13it/s][A[A
    
     88%|████████▊ | 598/682 [00:08<00:01, 65.14it/s][A[A
    
     89%|████████▊ | 605/682 [00:08<00:01, 66.02it/s][A[A
    
     90%|████████▉ | 612/682 [00:08<00:01, 66.75it/s][A[A
    
     91%|█████████ | 619/682 [00:09<00:00, 66.07it/s][A[A
    
     92%|█████████▏| 626/682 [00:09<00:00, 65.73it/s][A[A
    
     93%|█████████▎| 633/682 [00:09<00:00, 66.69it/s][A[A
    
     94%|█████████▍| 640/682 [00:09<00:00, 67.22it/s][A[A
    
     95%|█████████▌| 648/682 [00:09<00:00, 69.43it/s][A[A
    
     96%|█████████▌| 655/682 [00:09<00:00, 69.35it/s][A[A
    
     97%|█████████▋| 663/682 [00:09<00:00, 70.23it/s][A[A
    
     98%|█████████▊| 671/682 [00:09<00:00, 70.02it/s][A[A
    
    100%|█████████▉| 679/682 [00:09<00:00, 68.07it/s][A[A
    
    100%|█████████▉| 681/682 [00:09<00:00, 68.80it/s][A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidYellowLeft.mp4 
    
    CPU times: user 30.4 s, sys: 472 ms, total: 30.9 s
    Wall time: 10.2 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>





```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidYellowLeft.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```

## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```

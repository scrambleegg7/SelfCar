
# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

My final goals / steps of this 1st project of SelfCar driving are the following:
* Goal is Make and Confirm a pipeline that finds lane lines on the road.
* Using Python and OpenCV component.
* Use highly-advaned image logic like GrayScale Gaussina Hough etc of OpenCV module to handle data images effectively. 
* Finally output Stremlined Video Images to trace and show the lane line in video format.  

[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"
[gray1]: ./test_gray_images/solidWhiteCurve.jpg
[gauss1]: ./test_gauss_images/solidWhiteCurve.jpg
[canny1]: ./test_canny_images/solidWhiteCurve.jpg
[masked]: ./test_masked_images/solidWhiteCurve.jpg
[houghoriginal]: ./test_hough_originalline_images/solidWhiteCurve.jpg
[hough]: ./test_hough_images/solidWhiteCurve.jpg
[blend]: ./test_combine_images/solidWhiteCurve.jpg

---


### My Pipeline. 

My main pipeline of the project consisted of 5 steps as follows.

1. __Grayscale__
This is a first step to convert colored image into Black and White image to suppress any impact of 3 layers of colored image. Not only shrinking the data size / amount to single dimention, but also gives more advanced benefits for further image processing in next step explained.   

 ![Gray Scale][gray1]

2. __Gaussian Blur__
The second step is Gaussian Blur processing which remove high-frequency data (a.k.a noize) from images. It also gives completed image more smoother in data block.
With using simple convolutional technique to normarize image from the center of the selected block, I have setup kernel parameter = 15 and then generated from gray scale images as below. As a result, this process emphasized white lanes in the image compared with previous gray scaled image.


 ![Gaussian Blur image][gauss1]

3. __Canny__
Next step is canny image processing which generates from Gaussian Blur. It simply calculates the gradient of the image and draw the edge with black and white image. The generated image as follows;

 ![Canny image][canny1]

4. __Masked and Hough__
Next, I generatd Masked image with using helper function region of interest, whih selects only limited area with vertics. Other areas except vertics should be omiited for further detecting processing.
Area which I defined is hard-coded triangle one in where the center of area is set to vertical size of image divid by 2 + alpha. 

![masked][masked]



then apply hough helper function to detec the lane line from cannied image (step3)image. I have tuned and applied below parameters to hough helper function after spending time to find best combination of parameters. 

```
rho = 1  #3
theta = np.pi/180
threshold = 10
min_line_length = 30
max_line_gap = 1
```  
![masked][houghoriginal]

At first step to apply non-customized draw_line function, 2 lane lines are separatelly fragmented with several pieces. Either one of lines is not complete style as what we wanted to see as one straight line from the bottom of the image. 
Thus I have designed simple interpolate function (using polyfit) to extend lines.


![masked][hough]

5. __Weighted Image (Blending image)__
Finally, I have combined a original test image with hough image using weighted_img helper function. 

![masked][blend]



### 2. Identify potential shortcomings with your current pipeline

I think there are a few shortcomings on my current project

* region of interest is hard coded. I have setup fully hard-coded data on program module. Also, I have manually found which region parameter should be fitted in the image. It is not realisi 

* I have used many module layers starting from Gray Scale to hough helper function as program logic pipeline. In our test case where small data images are assigned, these logic will be satisified to get the final result to see the lane line on the road. Howere I am wondering if this approach could be ideal for the real world which needs more high resolution image data.  

### 3. Suggest possible improvements to your pipeline

I think following improvements would be necessary for further developments.

* To build the real time lane detection without manual adjustment to define the vertics area of the lane line, high-performance data logic would be necessarilly developed. 

* To cope with massive streamed image data, robust data logic to support scalability would be necessary.  

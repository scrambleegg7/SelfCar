# **Traffic Sign Recognition** 

## This is my Writeup

---

**Build a Traffic Sign Recognition Project (UdaCity)**

This is a second project assigned by UdaCity Self Car Driving Nano Degree Course. The main object is to classify the traffic Sign Board with tensorflow. Though the course provides free ticket to access Amazon cloud server (AWS) GPU environments, I will use my personal system environment having solid GPU card.

My personal system has;

* Ubuntu 16.04 TLS
* GTX 1080ti 
* 240GB SSD HDD  

I have each goals as following.



* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[trainhist]:./plotimage/train_label.png "train histogram"
[testhist]:./plotimage/test_label.png "test histogram"
[validhist]:./plotimage/valid_label.png "valid histogram"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### My Writeup Summary Report
***

### Data Set Summary & Exploration

#### 
* Data consists of 3 parts, training data and validation, test data image, those are packed into pickle data format. Thus I have determined to use numpy and pandas module to handle those data format, so that next image preprocessing can smoothly accept it.  

As a result of my further analysis of data exploration, I have found following overview of the data set:

#### 1. Train / Test / Validattion Data Summary ####

First of all, traffic signboard data set is:

* The size of training set is 34799
* The size of the validation set  4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (data size is shrinked.)
* All are colored image having 3 channels. (might need to change B&W image.)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Secondly, I have generated histogram against train, test, validation data set how label-id are distributed over those 3 types of signboard images. Looking at histgraph carefully, some of label id like SpeedLimit signboard has shown strong biases, which means having much volume of data than any other label id like trun left or turn right etc. Here is my visualiztion of the data set exploralization.   

![alt text][trainhist]
![alt text][testhist]
![alt text][validhist]



### Design and Test a Model Architecture

#### 1. GrayScale Preprocess
First of all, I have converted all colored images to gray scale using cv2 YCrBr method. The reason why I have choosed gray scape as first step is that I need to get rid out of any unnecessary colored cordination data and shrink dimention to 1, moreover to strength siginificant features of each images.

The below images are my random sample gray images.

![alt text][image2]

then I have applied the image scaling to normalize data set image, which aredivid with float number 255 (float number is necessary to get float result after deviding.). In some normalization cases, -0.5 is deducted after image is devided with 255., but I have used [0-1] normalization as standard image process. 
Also, these teqchnique, Gray Scale and Normalization process is conducted in tensorflow batch process after reading main augmented raw data.


#### 2. Data Augmentation Preprocess

Generally, data has to have good diversity as the object of interest needs to be present in varying sizes, lighting conditions and poses if desiring that our network generalizes well during training and testing phase. To overcome this problem of limited quantity and limited diversity of data for aquiring best performance of data training and testing, I have used data augmentation technique, where I modified 

#### 3. Increased Data size to fullfill imbalanced data size over traffic sign label.
As shown in the previous section, traffice sign data has strong data bias which means there are bunch of significant speed limit sign boards other than small data set of road construction sign. If we proceed to train twisted data volume per each label, then classifier will be out of right descision to determine undermined class label image, but also, return high volumed class label for unidentified images.
The below is result graph after balancing data.


![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 single image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Hyper Parameters on Tensorflow training.

To train the model, I used following hyper parameters:;

EPOCHS:64
learning_rate : 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 
* validation set accuracy of 0.96 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



# **Traffic Sign Recognition** 

## This is my Writeup

---

**Build a Traffic Sign Recognition Project (UdaCity)**

This is a second project assigned by UdaCity Self Car Driving Nano Degree Course. The main object is to have training data set of the traffic Sign Board with tensorflow and show accuracy performance for train / valid / test data set. For getting high-performance NVIDIA GPU processing, I have used my private system environment.

My personal system has;

* __Ubuntu 16.04 TLS__
* __GTX 1080Ti__ 
* __240GB SSD HDD__  

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
[signboard]:./signboard_img/label0-7.jpeg "label0-7 signboard"
[augmentation]:./plotimage/augmentation.png "augmentation"
[gray_scale]:./plotimage/gray_scale.jpeg "gray_scale"
[balanced_label]:./plotimage/balanced_label.png "balanced_label"
[DLImage]:./plotimage/DLImage.png "DownloadSign"
[DLImagePrediction]:./plotimage/DLImagePrediction.png "DLPrediction"
[DLImageSoftMax]:./plotimage/DLImageTop5Softmax.png "DLPredictionSoftMax"
[barChart]:./plotimage/barChart.png "barChart"
[pool1Features]:./plotimage/Pool1Features.png "Features"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

***
### My Writeup Summary Report
***

# 1. Data Set Summary & Exploration 
 
* Data consists of 3 main category **train** and **validation**, **test** image data set, which are packed into pickle format. I used numpy and pandas module to extract numpy data format from pickle files, so that next image preprocessing can smoothly accept it.  

As a result of further analysis of data exploration, I have found following overview of the data set:

## 1. Train / Test / Validattion Data Summary ####

First of all, traffic signboard data set is:

* The size of training set is 34799
* The size of the validation set  4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (data size is shrinked.)
* All are colored image having 3 channels. (might need to change B&W image.)
* The number of unique classes/labels in the data set is 43

## 2. Include an exploratory visualization of the dataset.

Secondly, I build histogram chart against train, test, validation data set how Traffic Sign labels are distributed over the whole data. In the overview of histgraph chart, some of label (eg. SpeedLimit 30km) has strong data biases, which means having much volume of data set than any other label (eg. trun left or turn right etc.). 
Here is my visualiztion of the data set exploralization.   

__Training Data__
![alt text][trainhist]
__Test Data__
![alt text][testhist]
__Validation Data__
![alt text][validhist]

The below is a part of raw traffic sign images.

![alt text][signboard]

# 2. Design and Test a Model Architecture

## Preprocessing 
### 2.1. GrayScale Preprocess
First of all, I have converted all colored images to gray scale using opencv YCrBr methodology. The reason why I have choosed to change gray scale is that I need to get rid out of any unnecessary colored cordination data and shrink dimention to 1, moreover to strength siginificant features of each images.
In special I have selected Y channel to feature Luminance component as described following brief explanation.

The below images are my random sample gray images.


##### __*About YCrCb*__ #####
>**Brief Note**
>The YCrCb color space is derived from the RGB color space and has the ?
>There are following three compoenents.
> 
>Y – Luminance or Luma component obtained from RGB after gamma correction.
>Cr = R – Y ( how far is the red component from Luma ).
>Cb = B – Y ( how far is the blue component from Luma ).
>
>This color space has the following properties.
>
>Separates the luminance and chrominance components into different channels.
>Mostly used in compression ( of Cr and Cb components ) for TV Transmission.
>Device dependent.
>
> **Observations**
>
>Similar observations can be made for Intensity and color components with regard to Illumination changes.
Perceptual difference between Red and Orange is less even in the outdoor image as compared to LAB.
White has undergone change in all 3 components.

**Result of Gray Image**

![alt text][gray_scale]

then I have applied the image scaling to normalize data set image, which are divided with float number 255. In some normalization cases, -0.5 is deducted to give center of image, but I have used [0-1] normalization as standard image processes. Normalized image data is saved into python class module variables as memory based.

### 2.2. Data Augmentation Preprocess

Generally, data has to have good diversity as the object of interest needs to be present in varying sizes, lighting conditions and poses during training and testing phase. To overcome this problem of limited quantity and limited diversity of data for aquiring best performance of data training and testing, I have applied augmentation technique, which modified image data which has combination of rotation, Shearing, and adjusting brightess. The below indicates a part of my program code.

```
        ang_rot = np.random.uniform(ang_range)-ang_range/2
        rows,cols,ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])

        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2

        # Brightness

        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

        shear_M = cv2.getAffineTransform(pts1,pts2)

        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))

        #if brightness == 1:
        #  img = augment_brightness_camera_images(img)

```
Additionally, I have blended "cutout" method as well as the above mentioned augmentation technique. Please see the reference document about ["Coutout".](https://arxiv.org/abs/1708.04552)

**Result of Augmentation**

![alt text][augmentation]

### 2.3. Increased Data size to fullfill imbalanced data size over traffic sign label.
As indicated in the previous section, traffice sign data has strong data bias which means there are mounts of speed limit sign boards other than small data set of road construction sign. If we proceed to train biased data, then final classifier model will be out of right descision to determine small amount of label image, also might return inaccurate results for unidentified images or tweeked images. 
To overcome biased data image, I have intentionally increased small amount of label image, so that all labels are equally distributed. 

![alt text][balanced_label]

### 2.4 Desgined Model Architecture 


#### Enhanced LeNet final Model used for training 

| Layer         		|     Description	        	| 
|:---------------------:|:---------------------------------------------:| 
| ***Input***          | 32x32x1 single image    		| 
| ***Convolution 1*** filter 5 x 5    | 1x1 stride, **same** padding, outputs: 32x32x32	|
| ReLu		|   ReLu(***Convolution1***)       			|
| ***Pooling1***	      	| 2x2 stride,  outputs 16x16x32 	|
| dropout	      	|  keep_prob = 0.9 |
| Input for Convolution2         | 16x16x32  (fromm MaxPooling1)   		| 
| ***Convolution2*** filer 5 x 5 |  1x1 stride, **same** padding, outputs: 16x16x64     		|
| ReLu		|   ReLue(***Convolution2***)       			|
| ***Pooling 2***	      	| 2x2 stride,  outputs 8x8x64 	|
| dropout	      	|  keep_prob = 0.8 |
| ***Convolution3*** filer 5 x 5 |  1x1 stride, **same** padding, outputs: 8x8x128     		|
| ReLu		|   ReLue(***Convolution3***)       			|
| ***Pooling 3***	      	| 2x2 stride,  outputs 4x4x128 	|
| dropout	      	|  keep_prob = 0.7 |
| ***Flatten1***		| Pooling(Pool1) Size4--> output 512 |
| ***Flatten2***		| Pooling(Pool2) Size2--> output 1024       |
| ***Flatten3***		| Input 4x4x128 --> output 2048       |
| ***Connecatenated***  | Flatten1 Flatten2 Flatten3  | 
| ***Fully Connected (fc3)***  | Input 3584 Output 1024  | 
| ***Fully Connected (fc4)***  | Input 1024 Output 43  | 
| logits		|   logits       		| 


### 3. Model Training

To train the model, I gave following hyper parameters to Lenet model;

EPOCHS:__32__
BATCH_SIZE: __128__
learning_rate : __0.0007__
Optimizer : __AdmOptimizer__
loss function : __tf.nn.softmax_cross_entropy_with_logits__ 


If an iterative approach was chosen:
* First of all, I have applied standard LeNet model (2 Convolutional Layer) architecture which accuracy score begins with 83%. With initial learning rate 0.001, accuracy score was gradually being improved upto 94% at final step of EPOCH. This is 1% score improvment than benchmark score, but it is not good enough to judge whether I am able to design best performance model for augmentation images. I start to think there might have any possibilities to enhance training model to preciously fit data set of aumentation images.   
* What I modified as main points are;
*** Use Batch Normalization Technique, but accuracy is not developed to show good performance than I expected. I have removed this function.  
*** add one additional layer (Convolutional layer increased to 3 layers)
*** increase filter number to 32 from convolutional layer 1
*** set Same instead of Valid on convolution layer
*** Add Dropout layer to set keep prob = 0.9 0.8 0.7
*** concantenate pool1 pool2 pool3 after max pooling for fully-connection layer  
* Due to huge size of Augmentation image data, I have made 30% test data from StratifiedShuffleSplit function. It is used to validate how training process accelarate to learn images with the convolutional layered model. 
* As a result of above modification on LeNet model, I was able to obtain best score 99% from training split data. (Please see 23 on Traffic_Sign html page.)

__Then I obtained accuracy score from final model__:
* training set accuracy of 0.99 (best performance from LeNet3)
* validation set accuracy of 0.95
* test set accuracy of .96


# 3. Test a Model on New Images

## 3.1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have setup several traffic sign images downloaed from web pages. I have picked up 24 different board images other than 5 images, since I would like to go through several images which have different shape of sign boards, also having many kinds of backgrounds behind the traffic sign.

![alt text][DLImage]

* *Before starting classification I have designed, personally I thought the model would have 50-60% accuracy for data set of raw images. The reason why I had so negative is that some of traffic signs are not correctly captured, but also they have many types of background images like curved roads in rural area and many types of weather conditions etc0. Thus, it would be hard to recognize these twisted image objects with neural networl models.* 

## 3.2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

See following prediction result from downloaded images. 
I have successfully generated right 84% accuracy score. But the score is less 16% than test score which I took in training processing. 

![alt text][DLImagePrediction]

The reason why I obtained 84% accuracy score from downloaded images is
* Downloaded Image size is not constant, some of larges, other small, in a result to resize images to 32 x 32 so that they fit to designed convolutional model. 
* Image is heavily distorted when shrinking big image to default entry image size (32x32). It means to drain some of significant data from original ones.
* Image is pictured with any kinds of background like road, building etc.
* Some Signboard is not pictured in center of image, thus hard to understand where traffic image is seen by machine learning model.

## 3.3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Regarding with softmax probabilities, see below top5 probabilities list for each predicted label from trained model. 
As observed in training section to show high accuracy score, Model trained with augmentation image data set performs well that it has 100% probability indicating exact same traffic sign as origin label (eg. ChildrenCrossing NoEntry) It cerntainly has designed to classify German Traffic Sign, however the model did not have enough cognition capability to classify several images like roadwork2. The model will be not deployed in real world.

![alt text][DLImageSoftMax]

also, I have build barChart figure which shows top5 softmax probabilities of predicted sign labels. 

![alt text][barChart]


# (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Yes, I have tried to extract features map from a top of convolutional layer, conv1, which has 16 x 16 dimention and size 32 filters panels. As seen in below screen image, those output images definitely display unique pattern which strength features of each images with sriding windowand max pooling technique.

![alt text][pool1Features]

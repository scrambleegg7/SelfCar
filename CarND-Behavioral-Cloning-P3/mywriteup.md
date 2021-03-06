# **Behavioral Cloning** 

## My Writeup report

---

**Behavioral Cloning Project**

It is 3rd project UdaCity setup for students attended in UdaCity Self Driving Car Engineer Term1 cource.
Project require to use keras for building trained model from splitted drive simulator JPG image data, then result to to drive autonomous drive simulator using training model.
The final goal is to learn keras neural network how to drive virtual car well in simulation cource, which first one is flat road having much left angle bias and second one is changing steepness of the terrain (up and down) where driver hardly can imagine forward road condition over the top of hill.
thus I have setup following specific goals step by step. 

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set 
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report 

To make goals, I used my main system as following environment stup;
Ubuntu 16.04 TLS
NVidia 1080i GTX
250GB SSD
Keras 2 under tensorflow 1.4

***

[//]: # (Image References)

[nvidia]: ./convmodel/nVidiaNet.png "Model Visualization"
[track1_center]: ./data/IMG/center_2016_12_01_13_30_48_287.jpg "track1 center"
[track1_left]: ./data/IMG/left_2016_12_01_13_30_48_287.jpg "track1 center"
[track1_right]: ./data/IMG/right_2016_12_01_13_30_48_287.jpg "track1 center"

[track2osx_center]: ./data_tr2/IMG/center_2018_03_04_06_20_44_422.jpg "track1 center"
[track2osx_left]: ./data_tr2/IMG/left_2018_03_04_06_20_44_422.jpg "track1 center"
[track2osx_right]: ./data_tr2/IMG/right_2018_03_04_06_20_44_422.jpg "track1 center"

[track2linx_center]: ./data_combined/IMG/center_2018_03_04_21_42_07_453.jpg "track1 center"
[track2linx_left]: ./data_combined/IMG/left_2018_03_04_21_42_07_453.jpg "track1 center"
[track2linx_right]: ./data_combined/IMG/right_2018_03_04_21_42_07_453.jpg "track1 center"

[loss_tr1]: ./convmodel/loss_figure_tr1_20180304.png "loss tr1"
[loss_tr2]: ./convmodel/loss_figure_tr2_20180304.png "loss tr2"
[loss_comb]: ./convmodel/loss_figure_combination_20180304.png "loss tr2"


# Rubric Points
## Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
# Files Submitted & Code Quality

## 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Due to require heavy check routine for image and angle data, and also incmoing input parameters based on track scenarios (track1 & track2), thus I have setup a series of python class modules which have unique function for building / spliting train / test image data and formatting augmentation image for training.  
* drive.py for driving the car in autonomous mode (this is nothing to change from original code.)
* **model.h5** containing a keras trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

## 2. Submission includes functional code

Using the Udacity provided simulator and original drive.py file, the car can be driven autonomously around the track1 and track2 by executing with following code
```sh
python drive.py convmodel/model.h5

and then start below command with different terminal.
./drStart.sh
```


## 3. Submission code is usable and readable

__Note__
keras recently released new version 2.x, which of some of main layers function have been completely changed to new format. Thus keras 1.2.1 format is not compatible to keras 2.x. I have placed 2 different version program codes on model.py and nVidiaModel.py, thus those can be seemlessly transferred each keras version. 
I have explained main program structure as follows;  

### a. model.py
model.py is a main program to call data streaming (DataObject class) and image processing (ImageObject), keras neural networl module.
It also has generator function which is used by keras fit_generator method.
Generator simply submits result with yield function every time batch sized queue is fully reached in small packets.

### b. Parameter.py
Parmeter.py encupsulates argparser class module. Main function is
* hold input parameters on class module.
* do sanitary check for some of incoming input parameters.
* set default parameter if none of paramters is defined with model.py

### c. DataObject.py
Its main object is to controll of reading file path of image data and drive control data (eg. steering, throtle, brake etc.) from driving_log.csv.
CSV file holds 3 different image file path for center, right and left angle views, thus I have desgined following points;
* use pandas read_csv to save each data into column, thus easy to pick up every view images for training purposes.
* split image path center, right and left image path. 
* adjust angles for right and left angle view, finally I have setup 2.1 after numerous trying to change adjusted value.
* show the statistics to display how many images and angles are read into class module. Pls. see the attached image.
* Train / Test data splitter (sklearn train_test_split function), then I have splitted 85% for train data and 15% for testing.
* (optional) : As optional function, I have setup to get rid out of several small angles from main data, which size can be defined with input parameter. Since so many left angle biased data are collected on track1 course, I would drop small angle and collect large angle.  


example of output
``` 
------------------------------
Exclude straight line images and angles from training / validation data..
Omit Angle less than 0.10
----------------------------------------
    total length of images / angles   
 Images - center:10830  left:10830  right:10830 
 Angles - center:10830  left:10830  right:10830 
original large center angle counts.. 1566
original large left angle counts.. 9182
original large right angle counts.. 404

original small center angle counts.. 9264
original small left angle counts.. 1648
original small right angle counts.. 10426

randomly picked up center angle counts.. 9264
randomly picked up left angle counts.. 1648
randomly picked up right angle counts.. 10426

Integrated Images - center:(10830,)  left:(10830,)  right:(10830,) 
Integrated Angles - center:(10830,)  left:(10830,)  right:(10830,) 
 Train / Test splitted size -->  (27616,) (4874,) (27616,) (4874,)
```
### d. ImageObject.py
This is a controller to process track image data, that is called from generator subroutine from model.py program. The main aim has following object; 
* to accept splitted training / test data as input parameters
* build batch segmentation data (default size = 32)
* read image data with file path, and image data is automatiically converted with skimage.io.imread method.
* for center angle image, select flip image with 50% random chance

### e. nViDiaModel.py
keras training model is provided in this class module
Main function and methods are; 
*  build neural network and reshape input image size to focus on front view of the camera.
*  embedded image preprocess function helps keras model only to grab truncated image area but also force to train with limited data size
*  speed up training time.
*  (Optional) define other style neural network setting Dropout layer for experimental trials to see impact of training loss.
I have designed those keras neural network model inspired by by [End-to-End document](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)   

# Model Architecture and Training Strategy


## 1. An appropriate model architecture has been employed

Inspired by nVidia network model, I have designed that Keras model has 5 convolutional layers, and 4 fully-connected layers.

| Layer         		|     Description	        	| 
|:---------------------:|:---------------------------------------------:| 
| ***Input***          | 90x320x3 single image (cropped image)    		| 
| ***Convolution 1*** filter 5 x 5    | 2x2 stride, **same** padding, outputs: 43x158x24	|
| ReLu		|   ReLu(***Convolution1***)       			|
| ***Convolution 2*** filter 5 x 5    | 2x2 stride, **same** padding, outputs: 20x77x36	|
| ReLu		|   ReLu(***Convolution2***)       			|
| ***Convolution 3*** filter 5 x 5    | 2x2 stride, **same** padding, outputs: 8x37x48	|
| ReLu		|   ReLu(***Convolution3***)       			|
| ***Convolution 4*** filter 5 x 5    | 2x2 stride, **same** padding, outputs: 6x35x64	|
| ReLu		|   ReLu(***Convolution4***)       			|
| ***Convolution 5*** filter 5 x 5    | 2x2 stride, **same** padding, outputs: 4x33x64	|
| **Flatten**		|   n x 8448      (n=batch size) 	|
| **Dense1**		|   n x 100       			|
| **Dense2**		|   n x 50       			|
| **Dense3**		|   n x 10       			|
| **Dense4**		|   n x 1       			|

## 2. Attempts to reduce overfitting in the model

One of way to minimize overfitting trend over the training data is to set Dropout after convolutional layer or set regularization parameters on each layers, however I have not taken any approach for giving designed model cutoff filters or penalarized paramters. 
However, I have frequently mixed training data and testing data so that none of sequential images is put in batch processing queue to avoid one-side bias images from training proccess. Please see the attached code how to setup shuffling. 

``` model.py
def generator(X, y, baseDir, batch_size=32, remove_straight_angle=None):

    num_samples = X.shape[0]
    cwd = os.path.join(os.getcwd(),baseDir)
    cwd = os.path.join(cwd,"IMG")


    while 1: # Loop forever so the generator never terminates
        

        #
        #    shuffle(samples)
        #
        X, y = shuffle(X, y)

        imageDataObject = ImageDataObjectClass(X,y,cwd, remove_straight_angle)

        for offset in range(0, num_samples, batch_size):

            X_train, y_train = imageDataObject.batch_next(offset,batch_size)

            yield X_train, y_train

```

``` ImageObject.py

    def batch_next(self,offset,BATCH_SIZE=32):

        image_samples = self.X[ offset:offset+BATCH_SIZE ]        
        angle_samples = self.y[ offset:offset+BATCH_SIZE ]

        images = []
        angles = []        
        is_flip = False
        for idx, ( image_sample, angle_sample ) in enumerate( zip( image_samples, angle_samples )  ):
            
            actual_filename = image_sample.split("/")[-1]
            name = os.path.join( self.cwd, actual_filename )
            
            if "center" in actual_filename: #  or "right" in actual_filename:
                is_flip = True
            else:
                is_flip = False

            drive_image = self.readImage(name)
            if np.random.random() < 0.5 and is_flip:
                #drive_image = cv2.flip(drive_image, 1)
                drive_image = np.fliplr(drive_image)

                angle_sample *= -1.

            images.append(drive_image)
            angles.append(angle_sample)
        
            #
            # flip image with 50% chance
            #
                

        return shuffle( np.array( images ), np.array( angles ) )

```

## 3. Model parameter tuning

I have tested 2 types of optimizer, one is Nadam, another one is Adam. Though Nadam was unstable to get tiny loss from entired model, I found that Adam was much stable optimizer model to minimize loss (mse) from several trial running.
Learning rate was set to 0.001
```
    optimizer = Adam(lr=params.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
```

## 4. Appropriate training data

At first step, I used standard track1 data, which is provided by UdaCity Course program. This effort can be utilized for building overall training methodology to run virtual car smoothly on the track1 without any issues. 
*** 

# Model Architecture and Training Strategy

## 1. Solution Design Approach

I have designed several keras neural network models for testing capavilities to minimize mean squared error.
First of all, I have tested model without dropout, which ignore any overfitting the trained data. 
Though there are lack of logical methodology to avoid overfitting and underfitting baised data, this pure normal model works well to show minimum loss result from training process. Personally, I think that its good expectation comes from image preprocessing of flipping center image and having focus area from center, left and right camera views, in addition to rectifier function so-called ReLu. 
This model solution apply to track1 and track2 cource. 


```
ReLu: The rectifier function is an activation function f(x) = Max(0, x) 
which can be used by neurons just like any other activation function, 
a node using the rectifier activation function is called a ReLu node. 
The main reason that it is used is because of how efficiently 
it can be computed compared to more conventional activation functions
like the sigmoid and hyperbolic tangent, 
without making a significant difference to generalisation accuracy. 
The rectifier activation function is used instead of 
a linear activation function to add non linearity to the network,
otherwise the network would only ever be able to compute 
a linear function.
```

### **About Dropout**
In some of neural network models, Dropout is highly recommended to set after convolutional layer as one of regularization technique, which main purpose forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. However it offends to keep minimize training / validation loss over the full training process. Thus it leads to that the model performance is worse than baseline performance.
The reason why I guess is that the network is small relative to training dataset, so that regularization does not work well to implement performance, and finally hurt overall model performance. Dropout will be utilized for heavy network architecture, and need more number of epochs to train data set.  

Epochs=25
Drop off small angle data (< 0.1) = NO
Final MSE = 0.0157


**About Relu performance**
```
2018-03-05 22:44:57.504667: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0)
215/215 [==============================] - 74s - loss: 0.0776 - val_loss: 0.0395
Epoch 2/25
215/215 [==============================] - 77s - loss: 0.0335 - val_loss: 0.0350
Epoch 3/25
215/215 [==============================] - 79s - loss: 0.0302 - val_loss: 0.0330
Epoch 4/25
215/215 [==============================] - 78s - loss: 0.0284 - val_loss: 0.0335
Epoch 5/25
215/215 [==============================] - 78s - loss: 0.0277 - val_loss: 0.0317
Epoch 6/25
215/215 [==============================] - 80s - loss: 0.0254 - val_loss: 0.0317
Epoch 7/25
215/215 [==============================] - 78s - loss: 0.0246 - val_loss: 0.0340
Epoch 8/25
215/215 [==============================] - 81s - loss: 0.0238 - val_loss: 0.0338
Epoch 9/25
215/215 [==============================] - 80s - loss: 0.0233 - val_loss: 0.0310
Epoch 10/25
215/215 [==============================] - 79s - loss: 0.0224 - val_loss: 0.0333
Epoch 11/25
215/215 [==============================] - 81s - loss: 0.0218 - val_loss: 0.0313
Epoch 12/25
215/215 [==============================] - 81s - loss: 0.0211 - val_loss: 0.0306
Epoch 13/25
215/215 [==============================] - 82s - loss: 0.0204 - val_loss: 0.0346
Epoch 14/25
215/215 [==============================] - 79s - loss: 0.0201 - val_loss: 0.0331
Epoch 15/25
215/215 [==============================] - 80s - loss: 0.0197 - val_loss: 0.0311
Epoch 16/25
215/215 [==============================] - 80s - loss: 0.0188 - val_loss: 0.0293
Epoch 17/25
215/215 [==============================] - 79s - loss: 0.0185 - val_loss: 0.0305
Epoch 18/25
215/215 [==============================] - 81s - loss: 0.0181 - val_loss: 0.0318
Epoch 19/25
215/215 [==============================] - 80s - loss: 0.0177 - val_loss: 0.0286
Epoch 20/25
215/215 [==============================] - 80s - loss: 0.0177 - val_loss: 0.0291
Epoch 21/25
215/215 [==============================] - 82s - loss: 0.0167 - val_loss: 0.0291
Epoch 22/25
215/215 [==============================] - 80s - loss: 0.0162 - val_loss: 0.0291
Epoch 23/25
215/215 [==============================] - 81s - loss: 0.0162 - val_loss: 0.0284
Epoch 24/25
215/215 [==============================] - 81s - loss: 0.0159 - val_loss: 0.0292
Epoch 25/25
215/215 [==============================] - 79s - loss: 0.0157 - val_loss: 0.0296
dict_keys(['val_loss', 'loss'])
```
***

#### 2. Final keras Model Architecture

As described in earlier step, the following architecture is what I finally desgined


![alt text][nvidia]

>Summary layout from keras module.

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2 (Conv2D)               (None, 20, 77, 36)        21636     
_________________________________________________________________
conv3 (Conv2D)               (None, 8, 37, 48)         43248     
_________________________________________________________________
conv4 (Conv2D)               (None, 6, 35, 64)         27712     
_________________________________________________________________
conv5 (Conv2D)               (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0

```

# Creation of the Training Set & Training Process

Besides the standard set of image data for track1 provided by UdaCity, I recorded following driving data by hand.

## a. Track1 
* __*How to make data*__
Track1 is a counter-clock wised road, where the virtual car is running in the loop course, there is not any special start point and end point.
Thus I took to record 3 laps of counter-wised round road and 1 lap of clock wise round road. Since standard set of images have lots of left biased, then I made the virtual car run on counter clocked round road to offset one-sided bias.


* __Image data__
>left image
![left image][track1_left] 
center image
![alt text][track1_center]
right image
![alt text][track1_right]


## b. Track2 OSX version
* __*How to make data*__
Track2 OSX is a mountain view road which has lots of uphill and downhill terrain along with tight curves. There is unlikely to give one sided bias angles on driving car, in this case there are many opportunities to create blended image having standard bell shape angle distribution.
thus I used to drive car on 2 laps of counter-wised round road. Also it is one way mountain road, when contninue driving the car, it approaches the wood barrier to stop the road. I have resersed the car direction and continue to run.

* __Image data__
>left image
![left image][track2osx_left] 
center image
![alt text][track2osx_center]
right image
![alt text][track2osx_right]


## c. Track2 Linux version
* __*How to make data*__
Opening track2 on linux platform, we can see another type of mountain view road cource, where people hardly imagine next road status behind tight corners and beyond the top of hill after climbing the slope. Along with such complex road conditions, the road course also has a variety of steepness terrain that is hard to drive car even by human driving skill. Also, the cource has special features, one of which is the center line deviding left lane and right lane separately. Ultimately it is ideal to drive the car in either side, eg. left or right side, however I have choiced to use centerline where car is driving along with.

* __Image data__
>left image
![left image][track2linx_left] 
center image
![alt text][track2linx_center]
right image
![alt text][track2linx_right]

***

## Angle Ajustments
Data points collected with the Drive simulator has only single steering angle, even though there are 3 different types of camera views. Therefore, manual adjutments for car steering angle have to get it done to accept proper angle data into neural network. As a result to run in my experimental trials, I have choosened follwing offset parameters.

| SIDE Camera         		|     Offset value	        	| 
|:---------------------:|:---------------------------------------------:| 
| **Left**          | +0.21    		| 
| **Right**          | -0.21    		| 


## Image Augmentation

* I did not take majour types of image augmetation like rotation, cutoff etc., but I flipped center images with 50% chance to make wide range image data distribution.
* Secondly, I have setup image normalization for every image data. This prerpcocess is cordinated by keras standard method eg. lambda. 
* Image is cut 50px from the top of the frame, and 20px from the bottom of the frame. 

Image augmentation can be viewed in ImageProcess.py 

```
            drive_image = self.readImage(name)
            if np.random.random() < 0.5 and is_flip:
                #drive_image = cv2.flip(drive_image, 1)
                drive_image = np.fliplr(drive_image)

                angle_sample *= -1.

```




# Training #

Those 3 types of driving scenes are put in the consolidated directory to learn every available correct driving angle, which is right choice of steering the handle for driving the virtual car. 


* __Split Train / Validation__
For the purposes of deminishing bias variant issues, I have splitted training data and validation data, which ratio 85% training and 15% validation for each scenario cases. Before starting to make batch process (batch generator), those splitted data are randomly shuffled.

* __Loss curve__
As described in earlier section, my training strategy is that the training has been done with 25 epochs and 0.001 learning rate for Adam Optimizer. As a result of this, I have finally obtained good result. Please see the attached loss graph.


>Trainig / Validation Loss figure
__Track1__
![alt text][loss_tr1]
__Track2 Linux__
![alt text][loss_tr2]
__Combination Track__
![alt text][loss_comb]

## Recovery Driving technique
Finally, I have mentioned how to make model learn recovery driving technique. Seeing **first 15-20 sec on track1_run.mp4**, car is flipping in the road every time it closes the edge of course, then car changes the steering angle to lead correct position of the car from the edge of the road. This correction technique is learnt by weaving the car in the traffic and out of the traffic. I intentionally change the direction to right one so that car can drive right lane and road away from the edges of the road.
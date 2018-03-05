# **Behavioral Cloning** 

## My Writeup report

---

**Behavioral Cloning Project**

It is 3rd project UdaCity setup for students attended in UdaCity Self Driving Car Engineer Term1 cource.
Project require to use keras for building trained model from splitted drive simulator JPG image data, then result to to drive autonomous drive simulator using training model.
The final goal is to learn keras neural network how to drive virtual car well in simulation cource, which first one is flat road having much left angle bias and second one is changing steepness of the terrain (up and down) where driver hardly can imagine forward road condition over the top of hill.
thus I have setup following specific goals step by step. 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior -> done
* Build, a convolution neural network in Keras that predicts steering angles from images -> done
* Train and validate the model with a training and validation set ->
* Test that the model successfully drives around track one without leaving the road -> not
* Summarize the results with a written report -> not

To make goals, my system environment
Ubuntu 16.04 TLS
NVidia 1080i GTX
250GB SSD
Keras 2 under tensorflow 1.4


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
# Files Submitted & Code Quality

# 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Due to require heavy check routine for image and angle data, and also setup input parameters based on track scenarios (track1 & track2), thus I have setup a series of python class module which has unique function for building / spliting train / test image data and formatting augmentation image for training.  
* drive.py for driving the car in autonomous mode (this is nothing to change from original one.)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

# 2. Submission includes functional code
Using the Udacity provided simulator and original drive.py file, the car can be driven autonomously around the track1 and track2 by executing with following code
```sh
python drive.py convmodel/model.h5

and then start below command with different terminal.
./drStart.sh
```


# 3. Submission code is usable and readable

### a. model.py
model.py is a main program to read data streaming and image processing, keras neural networl module.
It also has generator function which is used by keras fit_generator method.
Generator simply submits result with yield function every time batch sized queue is fully reached in small packets.
keras fit_genertor is similar to tfRecords queue process. 

### b. Parameter.py
Parmeter.py encupsels argparser class module. Main function is
* hold input parameters on class module.
* do sanitary check for some of input parameters.
* set default parameter if none of paramters is defined with model.py

### c. DataObject.py
It handles reading file path of image data and drive control data (eg. steering, throtle, brake etc.) from driving_log.csv.
CSV file holds 3 different image file path for center, right and left angle views, thus I have desgined following points;
* use pandas read_csv to save each data into column, thus easy to pick up data for using training.
* split image path center, right and left image path. 
* adjust angles for right and left angle view, finally I have setup 2.1 after numerous trying to change adjusted value.
* show the statistics to display how many images and angles are read into class module. Pls. see the attached image.
* Train / Test data splitter (sklearn train_test_split function), then I have splitted 85% for train data and 15% for testing.
* (optional) : As optional function, I have setup to get rid out of several small angles from main data, which size can be defined with input parameter. Since so many left angle biased data are collected on track1 course, I would drop small angle and collect large angle.  

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
* to receive splitted training / test data
* build batch sized data
* read image data with file path, and image data is automatiically converted with skimage.io.imread method.
* for center angle image, select flip image with 50% random chance

### e. nViDiaModel.py
keras training model is provided in this class module, where it builds neural network and reshape input image size to focus on front view of the camera.
Embedded image preprocess helps keras model to only grab selected image area but also force to train with limited data size, as a result to speed up making training module.
I have designed keras neural network model inspired by by [End-to-End document](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)   

### Model Architecture and Training Strategy


#### 1. An appropriate model architecture has been employed

Keras model has 5 convolutional layers, and 4 fully-connected layers.

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
| **Flatten**		|   n x 8448       			|
| **Dense1**		|   n x 100       			|
| **Dense2**		|   n x 50       			|
| **Dense3**		|   n x 10       			|
| **Dense4**		|   n x 1       			|

#### 2. Attempts to reduce overfitting in the model

One of decreasing overfitting model is to set Dropout after convolutional layer or set regularization parameters on each layers, however I have not taken any approach for giving designed model cutoff filters or penalarized paramters. 
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

#### 3. Model parameter tuning

I have tested 2 types of optimizer, one is Nadam, another one is Adam. Though Nadam was unstable to get tiny loss from entired model, I found that Adam was stable optimizer to minimize loss (mse) from several trial running.
Learning rate was set to 0.001
```
    optimizer = Adam(lr=params.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
```

#### 4. Appropriate training data

At first step, I used standard track1 data, which is provided by UdaCity Course program. This effort can be utilized for building overall training methodology to run virtual car smoothly on the track1 without any issues.  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have designed several keras neural network models for testing capavilities to minimize mean squared error.
First of all, I have tested model without dropout, which ignore any overfitting the trained data. 
Though there are lack of logical methodology to avoid overfitting and underfitting baised data, this pure normal model works well to show minimum loss result from training process. Personally, I think that its good expectation comes from image preprocessing of flipping center image and having focus area from center, left and right camera views. 
This model solution apply to track1 and track2 cource. 


Epochs=25
Drop off small angle data (< 0.1) = NO
Final MSE = 0.008



#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

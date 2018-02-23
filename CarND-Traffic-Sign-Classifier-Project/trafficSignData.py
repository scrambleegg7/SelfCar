# Load pickled data
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from PIL import Image
from skimage.transform import rescale, resize, rotate
from skimage.color import gray2rgb, rgb2gray

from tfRecordHandlerClass import tfRecordHandlerClass

def dataload():
    training_file = "train.p"
    train_aug_file = "train_aug.p"
    validation_file="valid.p"
    testing_file = "test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    with open(train_aug_file, mode='rb') as f:
        train_aug = pickle.load(f)

    return train,valid,test,train_aug

def features_label(in_data):

    X, y = in_data['features'], in_data['labels']
    print("features data and label",X.shape,y.shape)
    return X,y

#
#   Sign Board Data load model
#

class SignData(object):

    def __init__(self):

        self.train, self.valid, self.test, self.train_aug = dataload()

    def getTrainFeatures(self):
        print("loading train data....")
        X, y = features_label(self.train)
        return X,y

    def getTrainAugmentsFeatures(self):
        print("loading train augmentation data....")
        X, y = features_label(self.train_aug)
        return X,y


    def getTestFeatures(self):
        print("loading test data....")
        X, y = features_label(self.test)
        return X,y

    def getValidFeatures(self):
        print("loading valid data....")
        X, y = features_label(self.valid)
        return X,y

#
# Sign Board Image Processing
#


class SignImageClass():

    def __init__(self):

        signdata = SignData()
        self.X_train,self.y_train = signdata.getTrainFeatures()
        self.X_test,self.y_test = signdata.getTestFeatures()
        self.X_valid,self.y_valid = signdata.getValidFeatures()
        self.X_train_aug,self.y_train_aug = signdata.getTrainAugmentsFeatures()

    def imagePreprocessNormalize(self):

        print("<SignImageClass> Image Preprocess...")
        print("     train test valid ...")
        self.X_train = self.preprocessImages(self.X_train)
        self.X_test = self.preprocessImages(self.X_test)
        self.X_valid = self.preprocessImages(self.X_valid)

    def train_aug_data_length(self):
        return self.X_train_aug.shape[0]


    def train_data_length(self):
        return self.X_train.shape[0]

    def test_data_length(self):
        return self.X_test.shape[0]

    def valid_data_length(self):
        return self.X_valid.shape[0]

    def getValidPreprocess(self):
        return self.X_valid,self.y_valid

    def label_one_hot(self):

        pass

    def shuffle_train_aug(self):

        num_images = self.X_train_aug.shape[0]
        r = np.random.permutation(num_images)
        self.X_train_aug = self.X_train_aug[r]
        self.y_train_aug = self.y_train_aug[r]

    def shuffle_train(self):

        num_images = self.X_prep_train.shape[0]
        r = np.random.permutation(num_images)
        self.X_prep_train = self.X_prep_train[r]
        self.y_train = self.y_train[r]

    def batch_train_aug(self,offset=0,batch_size=64):

        features_batch = self.X_train_aug[offset:offset+batch_size]
        labels_batch = self.y_train_aug[offset:offset+batch_size]

        return features_batch,labels_batch

    def batch_train(self,offset=0,batch_size=64):

        features_batch = self.X_train[offset:offset+batch_size]
        labels_batch = self.y_train[offset:offset+batch_size]

        return features_batch,labels_batch

    def batch_valid(self,offset=0,batch_size=64):

        features_batch = self.X_valid[offset:offset+batch_size]
        labels_batch = self.y_valid[offset:offset+batch_size]

        return features_batch,labels_batch

    def batch_test(self,offset=0,batch_size=64):

        features_batch = self.X_test[offset:offset+batch_size]
        labels_batch = self.y_test[offset:offset+batch_size]

        return features_batch,labels_batch

    def getGrayScale(self,img):

        # About YCrCb
        # The YCrCb color space is derived from the RGB color space and has the following three compoenents.

        # Y – Luminance or Luma component obtained from RGB after gamma correction.
        # Cr = R – Y ( how far is the red component from Luma ).
        # Cb = B – Y ( how far is the blue component from Luma ).

        # This color space has the following properties.

        # Separates the luminance and chrominance components into different channels.
        # Mostly used in compression ( of Cr and Cb components ) for TV Transmission.
        # Device dependent.

        # Observations

        # Similar observations as LAB can be made for Intensity and color components with regard to Illumination changes.
        # Perceptual difference between Red and Orange is less even in the outdoor image as compared to LAB.
        # White has undergone change in all 3 components.

        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return np.resize(YCrCb[:,:,0], (32,32,1))

    def rgb2gray(self,rgb):

        r, g, b = rgb[:, :,:,0], rgb[:, :,:,1], rgb[:,:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def preprocessImages(self,images):
        
        gray_images = list(  map(lambda im:self.getGrayScale( im ) , images[:]   )     )
        gray_images = np.array(gray_images) / 255.0 

        return gray_images

    def preprocess_image(self,X):

        # convert to gray scale
        X = self.rgb2gray(X)
        #X = getGrayScale(X)

        # normalize with [0 1]
        X_norm = (X / 255.).astype(np.float32)
        X_norm = X_norm.reshape(  X_norm.shape + (1,) )

        return X_norm


    def convert_to_full_records_prep(self):

        tf_filenames = ["signtraffic_train.tfrecords", "signtraffic_test.tfrecords", "signtraffic_valid.tfrecords"]
        image_list = [self.X_prep_train, self.X_prep_test, self.X_prep_valid]
        label_list = [self.y_train, self.y_test, self.y_valid]

        tfRecordCls = tfRecordHandlerClass()
        for idx, (images, labels) in enumerate(zip(image_list, label_list)):

            tfRecordCls.convert_to_records( images, labels, tf_filenames[idx] )

    def convert_to_full_records(self):

        tf_filenames = ["trafficSign_train.tfRecords", "trafficSign_test.tfrecords", "trafficSign_valid.tfrecords"]
        image_list = [self.X_train, self.X_test, self.X_valid]
        label_list = [self.y_train, self.y_test, self.y_valid]

        tfRecordCls = tfRecordHandlerClass()
        for idx, (images, labels) in enumerate(zip(image_list, label_list)):

            tfRecordCls.convert_to_records( images, labels, tf_filenames[idx] )



# Load pickled data
import pickle

import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from PIL import Image
from skimage.transform import rescale, resize, rotate
from skimage.color import gray2rgb, rgb2gray

def dataload():
    training_file = "train.p"
    validation_file="valid.p"
    testing_file = "test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return train,valid,test

def features_label(in_data):

    X, y = in_data['features'], in_data['labels']
    print("features data and label",X.shape,y.shape)
    return X,y

#
#   Sign Board Data load model
#

class SignData(object):

    def __init__(self):
        self.train, self.valid, self.test = dataload()

    def getTrainFeatures(self):
        print("loading train data....")
        X, y = features_label(self.train)
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
        X_train,self.y_train = signdata.getTrainFeatures()
        X_test,self.y_test = signdata.getTestFeatures()
        X_valid,self.y_valid = signdata.getValidFeatures()

        self.X_prep_train = self.preprocessImages(X_train)
        self.X_prep_test = self.preprocessImages(X_test)
        self.X_prep_valid = self.preprocessImages(X_valid)


    def train_data_length(self):
        return self.X_prep_train.shape[0]

    def test_data_length(self):
        return self.X_prep_test.shape[0]

    def valid_data_length(self):
        return self.X_prep_valid.shape[0]

    def getValidPreprocess(self):
        return self.X_prep_valid,y_valid

    def label_one_hot(self):

        pass

    def shuffle_train(self):

        num_images = self.X_prep_train.shape[0]
        r = np.random.permutation(num_images)
        self.X_prep_train = self.X_prep_train[r]
        self.y_train = self.y_train[r]

    def batch_train(self,offset=0,batch_size=64):

        features_batch = self.X_prep_train[offset:offset+batch_size]
        labels_batch = self.y_train[offset:offset+batch_size]

        return features_batch,labels_batch

    def batch_valid(self,offset=0,batch_size=64):

        features_batch = self.X_prep_valid[offset:offset+batch_size]
        labels_batch = self.y_valid[offset:offset+batch_size]

        return features_batch,labels_batch

    def batch_test(self,offset=0,batch_size=64):

        features_batch = self.X_prep_test[offset:offset+batch_size]
        labels_batch = self.y_test[offset:offset+batch_size]

        return features_batch,labels_batch

    def getGrayScale(self,img):
        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return np.resize(YCrCb[:,:,0], (32,32,1))

    def rgb2gray(self,rgb):

        r, g, b = rgb[:, :,:,0], rgb[:, :,:,1], rgb[:,:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def preprocessImages(self,images):
        ret_array = []
        for img in images:
            g_img = self.getGrayScale(img)
            g_img = (g_img / 255).astype(np.float32)
            ret_array.append( g_img )


        return np.array(ret_array)

    def preprocess_image(self,X):

        # convert to gray scale
        X = self.rgb2gray(X)
        #X = getGrayScale(X)

        # normalize with [0 1]
        X_norm = (X / 255.).astype(np.float32)
        X_norm = X_norm.reshape(  X_norm.shape + (1,) )

        return X_norm











    def convert_to_records(self,IMAGE_SIZE=128):

        filename = self.tfrecFilename

        print("writing tfrecords....",filename)
        writer = tf.python_io.TFRecordWriter(filename)

        imagelistDicts = self.argCls.readImageList()

        for idx, (k,v) in enumerate( imagelistDicts.items() ):
            img = Image.open(k)
            #img = img.resize( (IMAGE_SIZE,IMAGE_SIZE) )
            img_np = np.array(img, dtype=np.float32)

            rows,cols,depth = img_np.shape

            image_raw = img_np.tostring()
            if idx % 500 == 0 and idx > 0:
                print("%d records processed .." % idx)
                print(rows,cols,depth)
                #print(image_raw)

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int( v )),
                'image_raw': _bytes_feature(image_raw)}))

            writer.write(example.SerializeToString())

        writer.close()
        print("writing done....")
        print("%d records written on tfrecords .." % idx )

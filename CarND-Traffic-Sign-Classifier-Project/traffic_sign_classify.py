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

#from lenet import LeNet
from lenet2 import LeNet


# TODO: Fill this in based on where you saved the training and testing data

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




class SignData(object):

    def __init__(self):
        self.train, self.valid, self.test = dataload()

    def getTrainFeatures(self):
        X, y = features_label(self.train)
        return X,y
    
    def getTestFeatures(self):
        X, y = features_label(self.test)
        return X,y
        
    def getValidFeatures(self):
        X, y = features_label(self.valid)
        return X,y


class SignImageClass():

    def __init__(self):

        signdata = SignData()
        X_train,self.y_train = signdata.getTrainFeatures()
        X_test,self.y_test = signdata.getTestFeatures()    
        X_valid,self.y_valid = signdata.getValidFeatures()

        self.X_prep_train = self.preprocess_image(X_train)
        self.X_prep_test = self.preprocess_image(X_test)        
        self.X_prep_valid = self.preprocess_image(X_valid)

    def train_data_length(self):
        return self.X_prep_train.shape[0]

    def test_data_length(self):
        return self.X_prep_test.shape[0]

    def valid_data_length(self):
        return self.X_prep_valid.shape[0]

    def shuffle_train(self):

        num_images = self.X_prep_train.shape[0]
        r = np.random.permutation(num_images)
        self.X_prep_train = self.X_prep_train[r]
        self.y_train = self.y_train[r]

    def batch_train(self,offset=0,batch_size=64):
        
        features_batch = self.X_prep_train[offset:offset+batch_size]
        labels_batch = self.y_train[offset:offset+batch_size]
        
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
        X = rgb2gray(X)
        
        # normalize with [0 1]
        X_norm = (X / 255.).astype(np.float32)
        X_norm = X_norm.reshape(  X_norm.shape + (1,) ) 
        
        return X_norm


def data_load():

    s = SignImageClass()

    return s

def train_procedure():

    BATCH_SIZE = 128
    # data image preparation
    sign_image = data_load()
    #features_batch,labels_batch = sign_image.batch_train()

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y_ = tf.placeholder(tf.int64, [None])
    y_one_hot = tf.one_hot(y_, depth=43, dtype=tf.float32)

    logits = LeNet(x,43)

    with tf.variable_scope("cost") as scope:
        #cost = tf.reduce_sum(tf.pow(pred_y - y_, 2))/(2*n_samples)
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
        cost = tf.reduce_mean(softmax)

    with tf.variable_scope("train") as scope:

        #train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize( cost )

    with tf.variable_scope("acc") as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        EPOCH = 2
        length_train_data = sign_image.train_data_length()
        for it in range(EPOCH):

            print("EPOCH", it)

            # data shuffle 
            sign_image.shuffle_train()

            #data,labels = mnist.train.next_batch(32)
            #sh = data.shape
            #data = data.reshape(sh[0],28,28,1 )
            #data = new_data.reshape( new_data.shape + (1,)  )            
            #print(data.shape)
            for offset in range(0,length_train_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_train(offset,batch_size=BATCH_SIZE)
            
                feeds = {x:features_batch, y_:labels_batch}
                train_op.run(feed_dict=feeds)
                #summary_str = sess.run(merged_summary_op, feed_dict=feeds)
                #summary_writer.add_summary(summary_str, it)

                if offset % 256 == 0:
                    acc, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                    print("step %d, accuracy:%.4f cost:%.4f"%(offset,acc,cost_))




def main():
    train_procedure()

if __name__ == "__main__":
    main()
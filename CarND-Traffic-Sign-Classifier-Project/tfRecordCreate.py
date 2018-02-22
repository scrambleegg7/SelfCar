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
from trafficSignData import SignImageClass

def readtfrecord():

    #signImageCls = SignImageClass()
    tfRecordCls = tfRecordHandlerClass()


    FILE = "./tfRecords/signtraffic_train.tfrecords"
    filename_queue = tf.train.string_input_producer([ FILE ], num_epochs=None)

    # get images labels from tf records
    images, labels = tfRecordCls.read_and_decode(filename_queue)

    #
    #   
    #

    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run( [init,init2]  )
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(2):
            img = sess.run(images)
            label = sess.run(labels)
            
            print(len(img))
            print(label)
            print(img.shape)

        coord.request_stop()
        coord.join(threads)
    print(img[0].ravel()) 
    sns.distplot(img[0].ravel()  )
    plt.show()


def createtfrecord():

    signImageCls = SignImageClass()

    #g_img = signImageCls.getGrayScale(  signImageCls.X_train[0]  )
    #sns.distplot( g_img.ravel() / 255. )
    #plt.show()
    signImageCls.convert_to_full_records()
    

def main():

    createtfrecord()
    #readtfrecord()



if __name__ == "__main__":
    main()



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
from ImageProcess import ImageProces

def readtfrecord():

    tfRecordCls = tfRecordHandlerClass()
    imgProcCls = ImageProces()



    FILE = "./tfRecords/trafficSign_aug.tfRecords"
    FILE = "./tfRecords/trafficSign_train.tfRecords"    
    filename_queue = tf.train.string_input_producer([ FILE ], num_epochs=4)

    # get images labels from tf records
    #images, labels = tfRecordCls.read_and_decode(filename_queue,BATCH_SIZE=100,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=124465)
    images, labels = tfRecordCls.read_and_decode(filename_queue,BATCH_SIZE=100,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=34799)

    #
    #   
    #

    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run( [init,init2]  )
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            #while True:
            local_steps = 0
            while not coord.should_stop():
                img,label = sess.run([images,labels])
                local_steps += 1
                #
                # if you want to apply gray scale from tfRecords...
                #
                img = list( map( lambda im : imgProcCls.getGrayScale(im)  , img[:] ) )
                img = np.array(img)
                #print(img.shape)

        except tf.errors.OutOfRangeError:
            # This will be raised when you reach the end of an epoch (i.e. the
            # iterator has no more elements).
            print("REACHED.   OutOfRange from EPOCH (string_input_producer)")
            print("  local steps --> ",local_steps)
            pass                 

        # Perform any end-of-epoch computation here.
        print('Done training, epoch reached')

        coord.request_stop()
        coord.join(threads)
    #print(img[0].ravel()) 
    #sns.distplot(img[0].ravel()  )
    #plt.show()
    #img = img.reshape( img.shape[0],img.shape[1],img.shape[2]   )
    #imgProcCls.displayImage(img)


def createtfrecord():

    signImageCls = SignImageClass()
    signImageCls.convert_to_full_records()
    

def main():

    #createtfrecord()
    readtfrecord()



if __name__ == "__main__":
    main()



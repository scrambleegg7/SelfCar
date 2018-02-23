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

from lenet2 import LeNet

def traing_testing():

    tfRecordCls = tfRecordHandlerClass()
    imgProcCls = ImageProces()





    FILE_train = "./tfRecords/trafficSign_aug.tfRecords"
    FILE_valid = "./tfRecords/trafficSign_test.tfRecords"    
    filename_queue_train = tf.train.string_input_producer([ FILE_train ], num_epochs=4)
    filename_queue_test = tf.train.string_input_producer([ FILE_test ], num_epochs=1)

    # get images labels from tf records
    images_train, labels_train = tfRecordCls.read_and_decode(filename_queue_train,BATCH_SIZE=100,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=124465)
    images_test, labels_test = tfRecordCls.read_and_decode(filename_queue_test,BATCH_SIZE=100,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=34799)

    #
    #  define LeNet Model   
    #
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

    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run( [init,init2]  )
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            local_steps = 0
            while not coord.should_stop():
                img,label = sess.run([images_train,labels_train])
                local_steps += 1
                #
                # if you want to apply gray scale from tfRecords...
                #
                img = list( map( lambda im : imgProcCls.getGrayScale(im)  , img[:] ) )
                img = np.array(img)
    
                feeds = {x:img, y_:label}
                train_op.run(feed_dict=feeds)

                if local_steps % 1244 == 0:
                    total_acc = []

                    for offset in range(0,length_valid_data,BATCH_SIZE):
                        features_batch,labels_batch = sign_image.batch_valid(offset,batch_size=BATCH_SIZE)
                        feeds = {x:features_batch, y_:labels_batch}
                        acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                        total_acc.append( acc_ * BATCH_SIZE )
                    accuracy_ = np.sum( total_acc ) / np.float(length_valid_data)
                    print("EPOCH:%d total accuracy : %.4f" % (it, accuracy_) )




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


def main():

    traing_testing()



if __name__ == "__main__":
    main()



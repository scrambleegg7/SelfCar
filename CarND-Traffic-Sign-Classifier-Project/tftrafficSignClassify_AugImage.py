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

    imgProcCls = ImageProces()
    sign_image = SignImageClass()

    BATCH_SIZE = 64

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

        saver = tf.train.Saver()

        EPOCH = 2
        length_train_data = sign_image.train_aug_data_length()
        length_valid_data = sign_image.valid_data_length()

        for it in range(EPOCH):

            #print("EPOCH", it)
            # data shuffle
            sign_image.shuffle_train_aug()
            for offset in range(0,length_train_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_train_aug(offset,batch_size=BATCH_SIZE)

                features_batch = list( map( lambda im : imgProcCls.getGrayScale(im)  , features_batch[:] ) )
                features_batch = np.array(features_batch) / 255.

                feeds = {x:features_batch, y_:labels_batch}
                train_op.run(feed_dict=feeds)


            total_acc = []
            for offset in range(0,length_valid_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_valid(offset,batch_size=BATCH_SIZE)

                features_batch = list( map( lambda im : imgProcCls.getGrayScale(im)  , features_batch[:] ) )
                features_batch = np.array(features_batch) / 255.

                feeds = {x:features_batch, y_:labels_batch}
                acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                total_acc.append( acc_ * BATCH_SIZE )
            accuracy_ = np.sum( total_acc ) / np.float(length_valid_data)
            print("EPOCH:%d total accuracy : %.4f" % (it, accuracy_) )

        save_path = saver.save(sess,"./lenet_model/lenet2_aug.ckpt", global_step=it)
        print("-" * 30 )
        print("-- Model saved in file: ", save_path )



def main():

    traing_testing()



if __name__ == "__main__":
    main()



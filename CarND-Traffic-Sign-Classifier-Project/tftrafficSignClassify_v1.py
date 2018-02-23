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
from trafficSignData import SignImageClass

# TODO: Fill this in based on where you saved the training and testing data

#
# VERSION 1 : 
#
# input data -> train data 
#               gray scale YCrBr
# implement Lenet model 2 
#  97% accuracy.
# Model saved under Lenet_model directory
#

def image_data_load():

    s = SignImageClass()
    return s

def train_procedure():

    #BATCH_SIZE = 64

    # 
    # OSX has no NVIDIA GPU, thus I setup small number of BATCH_SIZE
    #
    
    BATCH_SIZE = 5
    
    # data image preparation
    sign_image = image_data_load()
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

        EPOCH = 64
        length_train_data = sign_image.train_data_length()
        length_valid_data = sign_image.valid_data_length()

        for it in range(EPOCH):

            #print("EPOCH", it)
            # data shuffle
            sign_image.shuffle_train()
            for offset in range(0,length_train_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_train(offset,batch_size=BATCH_SIZE)

                feeds = {x:features_batch, y_:labels_batch}
                train_op.run(feed_dict=feeds)
                #summary_str = sess.run(merged_summary_op, feed_dict=feeds)
                #summary_writer.add_summary(summary_str, it)

                #if offset % 5120 == 0:
                #    print("step %d" % offset)
                #    acc, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                #    print("step %d, accuracy:%.4f cost:%.4f"%(offset,acc,cost_))


            total_acc = []
            for offset in range(0,length_valid_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_valid(offset,batch_size=BATCH_SIZE)
                feeds = {x:features_batch, y_:labels_batch}
                acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                total_acc.append( acc_ * BATCH_SIZE )
            accuracy_ = np.sum( total_acc ) / np.float(length_valid_data)
            print("EPOCH:%d total accuracy : %.4f" % (it, accuracy_) )

        save_path = saver.save(sess,"./lenet_model/lenet2.ckpt", global_step=it)
        #save_path = saver.save(sess, "/tmp/model.ckpt")
        print("-" * 30 )
        print("-- Model saved in file: ", save_path )


def main():
    train_procedure()

if __name__ == "__main__":
    main()

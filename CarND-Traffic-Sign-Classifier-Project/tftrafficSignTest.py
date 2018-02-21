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

def runtestData():
    # Create some variables.

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y_ = tf.placeholder(tf.int64, [None])
    y_one_hot = tf.one_hot(y_, depth=43, dtype=tf.float32)

    logits = LeNet(x,43)
    sign_image = SignImageClass()

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


    print("running test Image Data with saved tf model")
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)

        #EPOCH = 64
        BATCH_SIZE = 64
        length_train_data = sign_image.train_data_length()
        length_valid_data = sign_image.valid_data_length()
        length_test_data = sign_image.test_data_length()

        ckpt = tf.train.get_checkpoint_state('./lenet_model/')
        if ckpt: # if any model existed
            last_model = ckpt.model_checkpoint_path # path for last model checkpoints
            print("-" * 30)
            print("    saved model fould ", last_model)
            print("")
            print("    loading .... " + last_model)
            saver = tf.train.Saver()
            saver.restore(sess, last_model) # restore load model

            print("-" * 30)
            print("valid image data. data size->%d batch_size -> %d" % (length_valid_data,BATCH_SIZE) )
            total_acc = []
            total_cost = []
            for offset in range(0,length_valid_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_valid(offset,batch_size=BATCH_SIZE)
                feeds = {x:features_batch, y_:labels_batch}
                acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                total_acc.append( acc_ * BATCH_SIZE )
                total_cost.append( cost_ * BATCH_SIZE )

            accuracy_ = np.sum( total_acc ) / np.float(length_valid_data)
            loss_ = np.sum( total_cost ) / np.float(length_valid_data)
            print("VALID: total accuracy : %.4f  LOSS:%.4f" % ( accuracy_, loss_) )

            print("-" * 30)
            print("test image data. data size->%d batch_size -> %d" % (length_test_data,BATCH_SIZE) )
            total_acc = []
            total_cost = []
            for offset in range(0,length_test_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_test(offset,batch_size=BATCH_SIZE)
                feeds = {x:features_batch, y_:labels_batch}
                acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                total_acc.append( acc_ * BATCH_SIZE )
                total_cost.append( cost_ * BATCH_SIZE )

            accuracy_ = np.sum( total_acc ) / np.float(length_test_data)
            loss_ = np.sum( total_cost ) / np.float(length_test_data)
            print("TEST: total accuracy : %.4f  Loss:%.4f" % (accuracy_,loss_) )

        else: # if any tensor model is not found...
            init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
            sess.run(init_op)



def main():

    runtestData()

if __name__ == "__main__":
    main()

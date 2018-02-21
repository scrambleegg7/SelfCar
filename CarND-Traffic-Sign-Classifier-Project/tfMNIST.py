#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys, os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from skimage.transform import rescale, resize, rotate

import logging

from time import strftime, gmtime

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def LeNet(x, out_num=43):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, out_num), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(out_num))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


def proc2_letnetiii():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    data,labels = mnist.train.next_batch(32)

    data = np.arra (map( lambda x : x.reshape( 28,28 ) ,data ) )
    #expand = lambda x : 
    print(data)

def proc2_letnet():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 28,28,1])
    resized = tf.image.resize_images(x, (32, 32))

    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
     

    logits = LeNet(resized,10)

    with tf.variable_scope("cost") as scope:
        #cost = tf.reduce_sum(tf.pow(pred_y - y_, 2))/(2*n_samples)
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        cost = tf.reduce_mean(softmax)

    with tf.variable_scope("train") as scope:

        #train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize( cost )

    with tf.variable_scope("acc") as scope:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        for it in range(10000):

            data,labels = mnist.train.next_batch(32)
            sh = data.shape
            data = data.reshape(sh[0],28,28,1 )
            #data = new_data.reshape( new_data.shape + (1,)  )            
            #print(data.shape)
            feeds = {x:data, y_:labels}

            train_op.run(feed_dict=feeds)
            #summary_str = sess.run(merged_summary_op, feed_dict=feeds)
            #summary_writer.add_summary(summary_str, it)

            if it % 1000 == 0:
                feeds = {x:data, y_:labels}
                acc, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                print("step %d, accuracy:%.4f cost:%.4f"%(it,acc,cost_))

                #saver.save(sess,"./model/gradienTestModel%d.ckpt" % it, global_step=it)

    

def proc1():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])


    with tf.variable_scope("mnist") as scope:

        name = "conv1"
        with tf.variable_scope(name) as scope:
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            x_image = tf.reshape(x, [-1,28,28,1])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            #w_1 = tf.summary.histogram("W_conv1", W_conv1)
            #b_1 = tf.summary.histogram("b_conv1", b_conv1)


        name = "conv2"
        with tf.variable_scope(name) as scope:
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        name = "fc1"
        with tf.variable_scope(name) as scope:
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        name = "fc2"
        with tf.variable_scope(name) as scope:
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

        with tf.variable_scope("logits_pred") as scope:
            #logits = tf.matmul(x, W) + b
            #logits = tf.nn.relu(logits)
            logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"mnist")
    #print([v.name for v in var])
    conv1_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "mnist")[0]
    #conv2_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "mnist")[0]
    conv2_w= [v for v in tf.trainable_variables() if "conv2/W" in v.name][0]
    fc1_w= [v for v in tf.trainable_variables() if "fc1/W" in v.name][0]
    fc2_w= [v for v in tf.trainable_variables() if "fc2/W" in v.name][0]
    #Ws = [v for v in tf.trainable_variables() if "W" in v.name][0]
    # Create the model

    #w_2 = tf.summary.histogram("W_conv2", conv2_w)


    Ws = []
    for v in var:
        if "W" in v.name:
            print(v)
            Ws.append(v)

    n_samples = 32

    # Define loss and optimizer
    with tf.variable_scope("cost") as scope:
        #cost = tf.reduce_sum(tf.pow(pred_y - y_, 2))/(2*n_samples)
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        cost = tf.reduce_mean(softmax)

        tf.summary.scalar("cost", cost)

    with tf.variable_scope("train") as scope:
        grads = tf.gradients(cost, var)
        gradients = list(zip(grads, var))

        regularizer = 0.0
        for w in Ws:
            regularizer += tf.nn.l2_loss(w)
        #
        #print(gradients)
        beta = 0.01
        loss = tf.reduce_mean(cost + beta * regularizer)

        opt = tf.train.GradientDescentOptimizer(1e-4)
        #train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        g_and_v = opt.compute_gradients(cost, var)

        #p = 1.
        #eta = opt._learning_rate
        #my_grads_and_vars = [(g-(1/eta)*p, v) for g, v in grads_and_vars]
        train_op = opt.apply_gradients(grads_and_vars=g_and_v)

        for index, grad in enumerate(g_and_v):
            tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

    #with tf.variable_scope("train_gradients") as scope:
    #    grads = tf.gradients(cost, var)
    #    gradients = list(zip(grads, var))
        #print(gradients)
        #opt = tf.train.GradientDescentOptimizer(1e-4)
    #    opt = tf.train.AdamOptimizer(1e-4)     #.minimize(cross_entropy)
    #    g_and_v = opt.compute_gradients(cost, var)

        #p = 1.
        #eta = opt._learning_rate
        #my_grads_and_vars = [(g-(1/eta)*p, v) for g, v in grads_and_vars]
    #    train_op = opt.apply_gradients(grads_and_vars=g_and_v)

    with tf.variable_scope("Accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Merge all summaries into a single operator
    merged_summary_op = tf.summary.merge_all()
    save_path = None

    with tf.Session() as sess:

        if save_path is None:
            save_path = 'experiments/' + \
            strftime("%d-%m-%Y-%H:%M:%S/model", gmtime())
            print("No save path specified, so saving to", save_path)
        if not os.path.exists(save_path):
            logging.debug("%s doesn't exist, so creating" , save_path)
            os.makedirs(save_path)

        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(save_path, sess.graph)

        for it in range(10000):

            data,labels = mnist.train.next_batch(32)
            #print(data.shape, labels.shape)
            feeds = {x:data, y_:labels, keep_prob: 0.5}

            train_op.run(feed_dict=feeds)

            summary_str = sess.run(merged_summary_op, feed_dict=feeds)
            summary_writer.add_summary(summary_str, it)

            if it % 1000 == 0:
                feeds = {x:data, y_:labels, keep_prob: 1.}
                acc, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                print("step %d, accuracy:%.4f cost:%.4f"%(it,acc,cost_))

                saver.save(sess,"./model/gradienTestModel%d.ckpt" % it, global_step=it)


def weight_variable(shape):
    initial_value = tf.truncated_normal(shape, stddev=0.1)
    W = tf.get_variable("W",initializer=initial_value)
    return W

def bias_variable(shape):
    initial_value = tf.truncated_normal(shape, 0.0, 0.001)
    b = tf.get_variable("b",initializer=initial_value)
    return b

def conv2d(x, W, name="conv"):
    with tf.variable_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def checkGradVars(sess,grads_and_vars,feeds):

    gradients_and_vars = sess.run(grads_and_vars, feed_dict=feeds)
    for g, v in gradients_and_vars:
        if g is not None:
            print("****************this is variable*************")
            print("variable's shape:", v.shape)
            print(v)
            print("****************this is gradient*************")
            print("gradient's shape:", g.shape)
            print(g)

def main():

    proc2_letnet()




if __name__ == '__main__':
    main()
    #create_test_data()

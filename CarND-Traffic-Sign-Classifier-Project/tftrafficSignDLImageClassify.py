import pickle

import cv2
import os
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
from ImageProcess import ImageProces

def loadlabelnamelist():

    signnames = "signnames.csv"
    df_signname = pd.read_csv(signnames)

    signname_dict = {}
    for k,v in zip(df_signname["ClassId"].values,df_signname["SignName"].values):
        signname_dict[k] = v

    return signname_dict

def loadDownloadImage():

    imageProcCls = ImageProces()
    download_images = {}

    downloadDir = "./DownloadsSign"
    files = os.listdir(downloadDir)
    for name in files:
        filename = name.split(".")
        if filename[-1] != "jpg" and filename[-1] != "png":
            continue
        print(name)
        image = cv2.imread( os.path.join(downloadDir,name) )
        image = cv2.resize(image,(32,32))
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)    
        image = imageProcCls.getGrayScale(image) / 255.

        download_images[filename[0]] = image

    return download_images 

def runtestData():
    # Create some variables.

    download_images = loadDownloadImage()
    signname_dict = loadlabelnamelist()

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y_ = tf.placeholder(tf.int64, [None])
    y_one_hot = tf.one_hot(y_, depth=43, dtype=tf.float32)

    logits = LeNet(x,43)
    prob_img = tf.nn.softmax(logits)
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

            #
            # testing just 1 image 
            #
            
            acc = []
            for fname, img in sorted(download_images.items(), key=lambda x:x[0]):

                original_idx = int( fname.split("_")[-1] )
                img = img.reshape( (1,)+img.shape )
                probs_ = sess.run(prob_img, feed_dict = {x:img})
                #print(probs_)
                argidx = np.argmax(probs_)

                if argidx == original_idx:
                    acc.append(1)
                else:
                    acc.append(0)

                print(argidx, fname, signname_dict[argidx])

            print("model accuracy : %.4f " % np.mean( acc )  )


        else: # if any tensor model is not found...
            init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
            sess.run(init_op)



def main():

    #loadDownloadImage()
    runtestData()

if __name__ == "__main__":
    main()

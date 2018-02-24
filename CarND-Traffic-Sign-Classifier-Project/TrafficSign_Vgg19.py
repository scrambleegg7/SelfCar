#
import tensorflow as tf

from tfRecordHandlerClass import tfRecordHandlerClass
from trafficSignData import SignImageClass
from ImageProcess import ImageProces

import numpy as np

flags = tf.app.flags
DEFINE = flags.FLAGS

# Trainable version of VGG19

flags.DEFINE_float('random_normal_stddev', 1e-2, 'Random normal standard deviation')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')

flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('train_batch_size', 64, 'Training batch size')
flags.DEFINE_integer('validation_batch_size', 100, 'Validation batch size')

NUM_CONV1_FILTERS = 64
NUM_CONV2_FILTERS = 64
NUM_CONV3_FILTERS = 128
NUM_CONV4_FILTERS = 128
NUM_CONV5_FILTERS = 256
NUM_CONV6_FILTERS = 256
NUM_CONV7_FILTERS = 256
NUM_CONV8_FILTERS = 256
NUM_CONV9_FILTERS = 512
NUM_CONV10_FILTERS = 512
NUM_CONV11_FILTERS = 512
NUM_CONV12_FILTERS = 512
NUM_CONV13_FILTERS = 512
NUM_CONV14_FILTERS = 512
NUM_CONV15_FILTERS = 512
NUM_CONV16_FILTERS = 512
NUM_FC17_UNITS = 4096
NUM_FC18_UNITS = 4096
NUM_FC19_UNITS = 1000
NUM_FC19_UNITS_43 = 43

def linearize(x):
    x_shape = x.get_shape().as_list()
    x_length = x_shape[1] * x_shape[2] * x_shape[3]
    return tf.reshape(x, [-1, x_length]), x_length

def weight_and_bias(name, shape):
    with tf.variable_scope(name) as scope:
        _xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name=name+"_weights", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name=name+"_b", initializer=tf.zeros_initializer(shape=shape[-1]   )  )   
        #b = tf.get_variable(name=name+"_b", shape=shape[-1], initializer=tf.zeros_initializer(shape=shape[-1]   )  )
    return W, b

def weight_and_bias2(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=DEFINE.random_normal_stddev))
    b = tf.Variable(tf.truncated_normal([shape[-1]], stddev=DEFINE.random_normal_stddev))
    return W, b


def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')


class Vgg19(object):
    """
    A trainable version VGG19.
    """
    def __init__(self, trainable=True):
        self.trainable = trainable

    def build(self, rgb, dropout_keep_prob, NUM_CATEGORY=1000):
        #self.SKIP_LAYER = SKIP_LAYER
        logits = self.model(rgb,NUM_CATEGORY,dropout_keep_prob)
        return logits

    def model(self,x_2d, NUM_CLASS, dropout_keep_prob=0.5):

        IMAGE_SIZE_X = 256
        IMAGE_SIZE_Y = 256
        NUM_IMAGE_CHANNELS = 1

        #x_2d = tf.reshape(x_in, [-1, IMAGE_SIZE_X, IMAGE_SIZE_Y, NUM_IMAGE_CHANNELS])
        with tf.variable_scope("POOL1") as scope:
            w_conv1, b_conv1 = weight_and_bias("conv1",[3, 3, NUM_IMAGE_CHANNELS, NUM_CONV1_FILTERS])
            w_conv2, b_conv2 = weight_and_bias("conv2",[3, 3, NUM_CONV1_FILTERS, NUM_CONV2_FILTERS])
            conv1 = tf.nn.relu(tf.add(conv2d(x_2d, w_conv1), b_conv1))
            conv2 = tf.nn.relu(tf.add(conv2d(conv1, w_conv2), b_conv2))
            pool1 = max_pool(conv2)

        with tf.variable_scope("POOL2") as scope:
            w_conv3, b_conv3 = weight_and_bias("conv3",[3, 3, NUM_CONV2_FILTERS, NUM_CONV3_FILTERS])
            w_conv4, b_conv4 = weight_and_bias("conv4",[3, 3, NUM_CONV3_FILTERS, NUM_CONV4_FILTERS])
            conv3 = tf.nn.relu(tf.add(conv2d(pool1, w_conv3), b_conv3))
            conv4 = tf.nn.relu(tf.add(conv2d(conv3, w_conv4), b_conv4))
            pool2 = max_pool(conv4)

        with tf.variable_scope("POOL3") as scope:

            w_conv5, b_conv5 = weight_and_bias("conv5",[3, 3, NUM_CONV4_FILTERS, NUM_CONV5_FILTERS])
            w_conv6, b_conv6 = weight_and_bias("conv6",[3, 3, NUM_CONV5_FILTERS, NUM_CONV6_FILTERS])
            w_conv7, b_conv7 = weight_and_bias("conv7",[3, 3, NUM_CONV6_FILTERS, NUM_CONV7_FILTERS])
            w_conv8, b_conv8 = weight_and_bias("conv8",[3, 3, NUM_CONV7_FILTERS, NUM_CONV8_FILTERS])
            conv5 = tf.nn.relu(tf.add(conv2d(pool2, w_conv5), b_conv5))
            conv6 = tf.nn.relu(tf.add(conv2d(conv5, w_conv6), b_conv6))
            conv7 = tf.nn.relu(tf.add(conv2d(conv6, w_conv7), b_conv7))
            conv8 = tf.nn.relu(tf.add(conv2d(conv7, w_conv8), b_conv8))
            pool3 = max_pool(conv8)

        with tf.variable_scope("POOL4") as scope:
            w_conv9, b_conv9 = weight_and_bias("conv9",[3, 3, NUM_CONV8_FILTERS, NUM_CONV9_FILTERS])
            w_conv10, b_conv10 = weight_and_bias("conv10",[3, 3, NUM_CONV9_FILTERS, NUM_CONV10_FILTERS])
            w_conv11, b_conv11 = weight_and_bias("conv11",[3, 3, NUM_CONV10_FILTERS, NUM_CONV11_FILTERS])
            w_conv12, b_conv12 = weight_and_bias("conv12",[3, 3, NUM_CONV11_FILTERS, NUM_CONV12_FILTERS])
            conv9 = tf.nn.relu(tf.add(conv2d(pool3, w_conv9), b_conv9))
            conv10 = tf.nn.relu(tf.add(conv2d(conv9, w_conv10), b_conv10))
            conv11 = tf.nn.relu(tf.add(conv2d(conv10, w_conv11), b_conv11))
            conv12 = tf.nn.relu(tf.add(conv2d(conv11, w_conv12), b_conv12))
            pool4 = max_pool(conv12)

        with tf.variable_scope("POOL4") as scope:
            w_conv13, b_conv13 = weight_and_bias("conv13",[3, 3, NUM_CONV12_FILTERS, NUM_CONV13_FILTERS])
            w_conv14, b_conv14 = weight_and_bias("conv14",[3, 3, NUM_CONV13_FILTERS, NUM_CONV14_FILTERS])
            w_conv15, b_conv15 = weight_and_bias("conv15",[3, 3, NUM_CONV14_FILTERS, NUM_CONV15_FILTERS])
            w_conv16, b_conv16 = weight_and_bias("conv16",[3, 3, NUM_CONV15_FILTERS, NUM_CONV16_FILTERS])
            conv13 = tf.nn.relu(tf.add(conv2d(pool4, w_conv13), b_conv13))
            conv14 = tf.nn.relu(tf.add(conv2d(conv13, w_conv14), b_conv14))
            conv15 = tf.nn.relu(tf.add(conv2d(conv14, w_conv15), b_conv15))
            conv16 = tf.nn.relu(tf.add(conv2d(conv15, w_conv16), b_conv16))
            pool5 = max_pool(conv16)

        with tf.variable_scope("linear") as scope:
            linear, linear_length = linearize(pool5)

        with tf.variable_scope("fc17drop") as scope:
            w_fc17, b_fc17 = weight_and_bias("fc17",[linear_length, NUM_FC17_UNITS])
            fc17 = tf.nn.relu(tf.add(tf.matmul(linear, w_fc17), b_fc17))
            fc17 = tf.nn.dropout(fc17, dropout_keep_prob)

        with tf.variable_scope("fc18drop") as scope:
            w_fc18, b_fc18 = weight_and_bias("fc18",[NUM_FC17_UNITS, NUM_FC18_UNITS])
            fc18 = tf.nn.relu(tf.add(tf.matmul(fc17, w_fc18), b_fc18))
            fc18 = tf.nn.dropout(fc18, dropout_keep_prob)

        with tf.variable_scope("fc19drop") as scope:
            w_fc19, b_fc19 = weight_and_bias("fc19",[NUM_FC18_UNITS, NUM_FC19_UNITS])
            fc19 = tf.nn.relu(tf.add(tf.matmul(fc18, w_fc19), b_fc19))
            fc19 = tf.nn.dropout(fc19, dropout_keep_prob)

        with tf.variable_scope("consolidated") as scope:
            w_out, b_out = weight_and_bias("final",[NUM_FC19_UNITS, NUM_CLASS])

        with tf.variable_scope("logits") as scope:
            self.logits = tf.add(tf.matmul(fc19, w_out), b_out)
            
        return self.logits
        #return tf.nn.softmax()

    def cross_entropy(self,y_one_hot,logits):    
        with tf.variable_scope("cost") as scope:
            #cost = tf.reduce_sum(tf.pow(pred_y - y_, 2))/(2*n_samples)
            softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
            cost = tf.reduce_mean(softmax)
            return cost

    def train(self,cost):
        with tf.variable_scope("train") as scope:
            global_step = tf.Variable(0, name='global_step',trainable=False)

            #train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
            optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
            train_op = optimizer.minimize( cost )
            return train_op, global_step

    def accuracy(self, logits, y_one_hot):
        with tf.variable_scope("acc") as scope:
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy


def main(argv):

    print("Preparing training data.")
    #mnist = input_data.read_data_sets("./data/", validation_size=0, one_hot=True)

    print("-" * 30)
    print("      initialize resource")
    imgProcCls = ImageProces()
    sign_image = SignImageClass()
    sign_image.imagePreprocessNormalize()

    vgg19 = Vgg19()
    
    print("-" * 40)
    print("      defined hyper parameters")
    print("dropout keep prob:",  DEFINE.dropout_keep_prob)
    print("epochs :",  DEFINE.epochs)
    print("BATCH_SIZE :",  DEFINE.train_batch_size)

    NUM_CLASS = 43
    BATCH_SIZE = DEFINE.train_batch_size

    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    resized = tf.image.resize_images(x, (227, 227))

    #y_ = tf.placeholder(tf.float32, [None, NUM_CLASS])    
    y_ = tf.placeholder(tf.int64, [None])
    y_one_hot = tf.one_hot(y_, depth=NUM_CLASS, dtype=tf.float32)

    # placeholder drop out
    dropout_keep_prob = tf.placeholder(tf.float32)
        
    logits = vgg19.model(resized, NUM_CLASS , dropout_keep_prob)
    prob_img = tf.nn.softmax(logits)

    cost = vgg19.cross_entropy(y_one_hot,logits)
    train_op, global_step = vgg19.train(cost)
    accuracy = vgg19.accuracy(logits, y_one_hot)

    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run( [init,init2]  )
        saver = tf.train.Saver()

        length_train_data = sign_image.train_aug_data_length()
        length_valid_data = sign_image.valid_data_length()

        g_step=tf.train.global_step(sess, global_step)
            
        for it in range(DEFINE.epochs):

            sign_image.shuffle_train_aug()

            for offset in range(0,length_train_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_train_aug(offset,batch_size=BATCH_SIZE)

                #features_batch = list( map( lambda im : imgProcCls.getGrayScale(im)  , features_batch[:] ) )
                #features_batch = np.array(features_batch) / 255.

                feeds = {x:features_batch, y_:labels_batch , dropout_keep_prob:DEFINE.dropout_keep_prob   }
                train_op.run(feed_dict=feeds)

                print(g_step)
            #print('global_step: %s' % g_step)


            total_acc = []
            for offset in range(0,length_valid_data,BATCH_SIZE):
                features_batch,labels_batch = sign_image.batch_valid(offset,batch_size=BATCH_SIZE)

                #features_batch = list( map( lambda im : imgProcCls.getGrayScale(im)  , features_batch[:] ) )
                #features_batch = np.array(features_batch) / 255.

                feeds = {x:features_batch, y_:labels_batch, dropout_keep_prob:DEFINE.dropout_keep_prob}
                acc_, cost_ = sess.run([accuracy,cost],feed_dict=feeds)
                total_acc.append( acc_ * BATCH_SIZE )
            accuracy_ = np.sum( total_acc ) / np.float(length_valid_data)
            print("EPOCH:%d validation - total accuracy : %.4f" % (it, accuracy_) )




        #length_train_data = sign_image.train_aug_data_length()

        #sign_image.shuffle_train_aug()

        #features_batch,labels_batch = sign_image.batch_train_aug(0,batch_size=BATCH_SIZE)

        #print("input image shape.....", features_batch.shape)

        #logits_ = sess.run(logits,  feed_dict= {x:features_batch, dropout_keep_prob:DEFINE.dropout_keep_prob   })
        #prob_img = sess.run(prob_img,  feed_dict= {x:features_batch, dropout_keep_prob:DEFINE.dropout_keep_prob   })
        #print(logits_.shape)
        #y_label = np.argmax(prob_img, axis = 1)
    
        #y_train_true = labels_batch
        #print("    label true value...")
        #print(y_train_true.shape)
        #print(y_train_true)
    
        #print("    generated value...")
        #print(y_label.shape)
        #print(y_label)



if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf 
import numpy as np  
import pandas as pd 
import os

pio = tf.python_io
# 
#  helper function set int and bytes.
#
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class tfRecordHandlerClass(object):

    def __init__(self):


        # images --> n x h x w x d (eg. 3000, 32, 32, 3 etc.)
        # labels --> n
        # labels is not one hot code 
        self.test = False
        #self.images = images
        #self.labels = labels

    def read_and_decode(self,filename_queue,BATCH_SIZE=32,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=34799):

        # load data normal case

        #
        # Ensure that the random shuffling has good mixing properties.
        #
        min_fraction_of_examples_in_queue = 0.99
        min_queue_examples_ = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
        
        capacity_ = min_queue_examples_ + 3 * BATCH_SIZE

        #
        # TFRecordReader has GZIP option to compress 
        # Do not forget to set GZIP , otherwise tensorflow does not tell you what is wrong.os
        #
        reader_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        reader = tf.TFRecordReader(options=reader_option)

        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'height': tf.FixedLenFeature( [], tf.int64 ),
                        'width': tf.FixedLenFeature([], tf.int64),
                        'depth': tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                    })

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        # tf pack is discontinued function ....
        # rename to stack
        imshape = tf.stack( [ height, width, depth] )
        image = tf.decode_raw(features['image_raw'], tf.uint8)  # tf int32 --> accept decimal

        IMAGE_SIZE = 32
        DIMENTION = 3
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
        #image = tf.reshape(image, imshape)
        
        #
        # if original tfRecords stored with uint8 (0-255), below 2 lines are not necessay 
        # otherwise, image is wrongly dislayed.
        #
        #image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        #image = tf.cast(image, tf.float32)
        #image = image / 255. - 0.5
        #

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        
        images, labels = tf.train.shuffle_batch(
            [image, label],
            allow_smaller_final_batch=True,
            num_threads=4,
            batch_size = BATCH_SIZE,
            min_after_dequeue = min_queue_examples_,
            capacity = capacity_)

        return images , labels

    def convert_to_records(self, images, labels, tf_filename="TFRecordsSign"):

        filename = tf_filename
        print("writing tfrecords.... under tfRecords dir. ",filename)
        #writer = pio.TFRecordWriter(filename)

        filename = os.path.join("./tfRecords",filename)

        #
        # use compress version for saving hunk size of numpy records 
        #
        writer = pio.TFRecordWriter(
            filename, options=pio.TFRecordOptions(pio.TFRecordCompressionType.GZIP))
        # = self.images.shape[0]
        length_images,rows,cols,depth = images.shape
        print("writing record count ....",length_images)
        print("saving image shape ....",rows,cols,depth)
        
        for idx, image_id in enumerate( range(length_images) ):

            image_raw = images[idx].tostring()
            label = labels[idx]
            if idx % 10000 == 0 and idx > 0:
                print("%d records processed .." % idx)

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int( label )),
                'image_raw': _bytes_feature(image_raw) 
                }
                )
                )

            writer.write(example.SerializeToString())

        writer.close()
        print("writing done....")
        print("%d records written on tfrecords .." % (idx+1) )

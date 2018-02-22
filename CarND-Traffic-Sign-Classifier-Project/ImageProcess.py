# Load pickled data
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from PIL import Image
from skimage.transform import rescale, resize, rotate
from skimage.color import gray2rgb, rgb2gray
from skimage import transform, filters, exposure



from tfRecordHandlerClass import tfRecordHandlerClass

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params,output_shape=(32,32), mode=mode)

class ImageAugmentationClass(object):

    def __init__(self,test=False):
        self.test = test

    def choiceAugmentation(self,img):

        self.choice_flag = np.random.randint(0, 2, [1, 10]).astype('bool')[0]

        img = self.horizontal_flip(img)
        img = self.horizontal_flip(img)
        return img

    def horizontal_flip(self, image):
        if self.choice_flag[0]:
            image = image[:,::-1,:]
        return image
    def vertical_flip(self, image):
        if self.choice_flag[1]:
            image = image[::-1,:,:]
        return image


    def transform_image(self,img,ang_range,shear_range,trans_range,brightness=0):

        #Advantage function is prepared to show the picture of the same sign from different angles.
        #I have used blended function with using openCV Affine translations and numpy,
        # rotations, translations and shearing parameters should be used.
        # Rotation
        ang_rot = np.random.uniform(ang_range)-ang_range/2
        rows,cols,ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])

        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2

        # Brightness

        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

        shear_M = cv2.getAffineTransform(pts1,pts2)

        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        img = cv2.warpAffine(img,Trans_M,(cols,rows))
        img = cv2.warpAffine(img,shear_M,(cols,rows))

        #if brightness == 1:
        #  img = augment_brightness_camera_images(img)

        return img


    def random_crop(self,image, crop_size=(32,32)):

        h, w, _ = image.shape
        # determine top left from crop size
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])

        # from top, left, size should be calculated plus random (0-32)
        bottom = top + crop_size[0]
        right = left + crop_size[1]

        image = image[top:bottom, left:right, :]
        return image



class ImageProces(object):

    def __init__(self):

        data = self.load_data()
        self.X_train, self.y_train = data['features'], data['labels']
        print("train features and label..",self.X_train.shape,self.y_train.shape)

        self.n_train = self.X_train.shape[0]
        self.image_shape = self.X_train.shape[1:2]
        self.n_classes = len(set(self.y_train))

        #self.imagePreprocess()

    def load_data(self):

        training_file = "train.p"
        validation_file="valid.p"
        testing_file = "test.p"

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)

            return train
        print("load train data - ImageProces...")

    def imagePreprocess(self):

        ts, imgs_per_sign   = np.unique(self.y_train, return_counts=True)
        avg_per_sign        = np.ceil(np.mean(imgs_per_sign)).astype('uint32')

        print(ts)
        print(imgs_per_sign)
        print("Average Counts per each TrafficSign...", avg_per_sign)

        separated_data = []
        for traffic_sign in range( len(ts) ):
            images_in_this_sign = self.X_train[self.y_train == traffic_sign, ...]
            separated_data.append(images_in_this_sign)

        new_effect_cnt = list( map(lambda x: (3. * (avg_per_sign  / x) ).astype(np.int32)  , (imgs_per_sign)  ) )

        imgAugCls = ImageAugmentationClass()
        for idx, (loop_cnt, sign_images) in enumerate( zip(new_effect_cnt, separated_data)):
            print("label:%d  loop_cnt:%d total images (incl. augmented images.):%d " % (idx, loop_cnt, (loop_cnt * imgs_per_sign[idx]))  )
            print(sign_images.shape)
            for cnt in range(loop_cnt):
                X_aug_img = list( map( lambda image: imgAugCls.transform_image( image,20,10,5  ) , sign_images[:]  ) )
                X_aug_img = np.array(X_aug_img)


        return X_aug_img



def main():
    imgProcCls = ImageProces()
    X_aug_img = imgProcCls.imagePreprocess()

    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
    plt.figure(figsize=(12,12))
    for i in range(100):
        ax1 = plt.subplot(gs1[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        #img = transform_image(image,20,10,5,brightness=1)

        plt.subplot(10,10,i+1)
        plt.imshow(X_aug_img[i])
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()

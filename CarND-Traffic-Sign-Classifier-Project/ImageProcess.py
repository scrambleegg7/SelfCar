# Load pickled data
import pickle
import sys

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

        self.imgAugCls = ImageAugmentationClass()

    def data_initialization(self):

        data = self.load_data()
        self.X_train, self.y_train = data['features'], data['labels']
        print("[ImageProcess] train features and label..",self.X_train.shape,self.y_train.shape)

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

    def getGrayScale(self,img):

        # About YCrCb
        # The YCrCb color space is derived from the RGB color space and has the following three compoenents.

        # Y – Luminance or Luma component obtained from RGB after gamma correction.
        # Cr = R – Y ( how far is the red component from Luma ).
        # Cb = B – Y ( how far is the blue component from Luma ).

        # This color space has the following properties.

        # Separates the luminance and chrominance components into different channels.
        # Mostly used in compression ( of Cr and Cb components ) for TV Transmission.
        # Device dependent.

        # Observations

        # Similar observations as LAB can be made for Intensity and color components with regard to Illumination changes.
        # Perceptual difference between Red and Orange is less even in the outdoor image as compared to LAB.
        # White has undergone change in all 3 components.

        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return np.resize(YCrCb[:,:,0], (32,32,1))

    def rgb2gray(self,rgb):

        r, g, b = rgb[:, :,:,0], rgb[:, :,:,1], rgb[:,:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def imagePreprocess(self):

        ts, imgs_per_sign   = np.unique(self.y_train, return_counts=True)
        avg_per_sign        = np.ceil(np.mean(imgs_per_sign)).astype('uint32')

        print(ts)
        print(imgs_per_sign)
        print("Average Counts per each TrafficSign...", avg_per_sign)

        separated_data = []
        for label_index in ts:
            images_in_this_sign = self.X_train[self.y_train == label_index, ...]
            separated_data.append(images_in_this_sign)

        new_effect_cnt = list( map(lambda x: (3. * (avg_per_sign  / x) ).astype(np.int32)  , (imgs_per_sign)  ) )

        whole_aug_image_dict = {}
        new_y_train = []

        for idx, (loop_cnt, sign_images) in enumerate( zip(new_effect_cnt, separated_data)):
            print("label:%d  loop_cnt:%d new augmentation images.:%d " % (ts[idx], loop_cnt, (loop_cnt * imgs_per_sign[idx]))  )
            print("original image shape", sign_images.shape)
            separate_aug_image = sign_images.copy()
        
            for cnt in range(loop_cnt):
                X_aug_img = list( map( lambda image: self.imgAugCls.transform_image( image,30,5,5  ) , sign_images[:]  ) )
                X_aug_img = np.array(X_aug_img)

                separate_aug_image = np.vstack( (separate_aug_image,X_aug_img))

            # save whole images per label into dict
            whole_aug_image_dict[ ts[idx]  ] = separate_aug_image

            len_separate_aug_image = separate_aug_image.shape[0]
            extend_labels = np.full( len_separate_aug_image,  ts[idx]   )
            new_y_train.extend(extend_labels)            
            print("final shape (augment image + original image) per label" , separate_aug_image.shape  )

        ts, imgs_per_sign   = np.unique(new_y_train, return_counts=True)
        print("-"*30)
        print(" new y_train (label) count ..")
        print(ts)
        print(imgs_per_sign)

        all_aug_images = whole_aug_image_dict[ 0  ] # get first label image data
        for k,v in whole_aug_image_dict.items():
            if k == 0: # skip label 0, because it has already saved 
                continue
            all_aug_images = np.vstack( (all_aug_images, v) )

        assert(len(new_y_train) == all_aug_images.shape[0]  )

        print("new image data shape after augmentation." , all_aug_images.shape[0] )
        return all_aug_images, np.array( new_y_train )

    def displayImage(self,images):

        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
        plt.figure(figsize=(10,10))
        for i in range(100):
            ax1 = plt.subplot(gs1[i])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            #img = transform_image(image,20,10,5,brightness=1)

            plt.subplot(10,10,i+1)
            plt.imshow(images[i])
            plt.axis('off')

        plt.show()

    def displayImageDistribution(self,images):

        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
        plt.figure(figsize=(10,10))
        for i in range(100):
            ax1 = plt.subplot(gs1[i])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            #img = transform_image(image,20,10,5,brightness=1)

            plt.subplot(10,10,i+1)
            sns.distplot(images[i].ravel() )
            plt.axis('off')

        plt.show()

    def saveImagesPickle(self,new_train_data):

        bytes_out = pickle.dumps(new_train_data)
        max_bytes = 2**31 - 1
        n_bytes   = sys.getsizeof(bytes_out)

        output_path   = 'train_aug.p'
        with open(output_path, 'wb') as f:
            for idx in range(0, n_bytes, max_bytes):
                f.write(bytes_out[idx:idx+max_bytes])


def main():
    imgProcCls = ImageProces()
    imgProcCls.data_initialization()

    X_aug_img, y_train = imgProcCls.imagePreprocess()


    print("-"*30)
    print("creating pickle train augmentation ...")

    new_train_data = {
            'features': X_aug_img,
            'labels': y_train
        }    

    imgProcCls.saveImagesPickle(new_train_data)

    #tfRecordCls = tfRecordHandlerClass()
    #tfRecordCls.convert_to_records( X_aug_img, y_train, "trafficSign_aug.tfRecords" )



if __name__ == "__main__":
    main()

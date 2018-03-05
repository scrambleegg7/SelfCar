import numpy as np  
import pandas as pd 
import cv2
import os

from skimage.io import imread

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

class DataObjectClass():

    def __init__(self):
        pass

    def shuffleSplit(self,test_size=0.3):

        X = self.images
        y = self.steerings
        
        X, y = shuffle(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

        return X_train, X_test, y_train, y_test
        
    def loadCSVData(self, baseDir="data", filename="driving_log.csv", sample = True, straight_angle=None):

        # load 3 angle camera view from flat file
        
        # center
        # left 
        # right 
        #
        # setup directory
        #
        #
        if sample:
            cwd = os.path.join( os.getcwd(), baseDir )
            driving_log = os.path.join(cwd, filename)
            df_drive = pd.read_csv(driving_log)
        else:
            driving_log = os.path.join(baseDir,filename)
            df_drive = pd.read_csv(driving_log,header=None,names=["center","left","right","steering","throttle","brake","speed"])

        # read csv file to save drive data
        centerImages = np.array( df_drive["center"].tolist() ) #[:,np.newaxis]
        leftImages = np.array( df_drive["left"].tolist() )     #[:,np.newaxis]
        rightImages = np.array( df_drive["right"].tolist() )   #[:,np.newaxis]

        #
        #  Mar. 03 2018 changed 0.275 to 0.25
        #               min max function used for left and right angle)
        #offset = 0.25
        offset = 0.21
        #   left_angle = min(1.0, center_angle + 0.25)

        Steering_center = np.array( df_drive["steering"].tolist() )    
        Steering_left  = np.array( list( map(lambda steer:(steer + offset), Steering_center.copy() ) )  )  
        Steering_right = np.array( list( map(lambda steer:(steer - offset), Steering_center.copy() ) ) )  

        if straight_angle is not None:
            print("Exclude straight line images and angles from training / validation data..")
            print("Omit Angle less than %.2f" % straight_angle )

        print("-"* 40)
        print("    total length of images / angles   ")
        print(" Images - center:%s  left:%s  right:%s " % ( len(centerImages), len(leftImages), len(rightImages)  )  )
        print(" Angles - center:%s  left:%s  right:%s " % ( len(Steering_center), len(Steering_left), len(Steering_right)  )  )
        

        #
        # remove straight angle
        #
        # make mask for greater than angle parameter
        #
        mask_center = Steering_center > straight_angle
        mask_left = Steering_left > straight_angle
        mask_right = Steering_right > straight_angle

        print("original large center angle counts..", np.sum(mask_center))
        print("original large left angle counts..", np.sum(mask_left)   )
        print("original large right angle counts..", np.sum(mask_right)   )
        print("")
        # mask for less than angle parameter
        mask_center_small_angle = (mask_center == False)
        mask_left_small_angle = (mask_left == False)
        mask_right_small_angle = (mask_right == False)        



        #
        # actual index number for less than angle parameter eg. 1, 21, 35 etc.
        # index number corresponds to small angle masking data
        #
        number_index_less_than_angle_center = np.arange(len(mask_center))[mask_center_small_angle]
        number_index_less_than_angle_left = np.arange(len(mask_left))[mask_left_small_angle]
        number_index_less_than_angle_right = np.arange(len(mask_right))[mask_right_small_angle]

        print("original small center angle counts..", len(number_index_less_than_angle_center))
        print("original small left angle counts..", len(number_index_less_than_angle_left))
        print("original small right angle counts..", len(number_index_less_than_angle_right))
        print("")
        #
        # next pickup random choice by shrink rate 
        shrink_rate = 1. 
        #
        new_index_center = np.random.choice(number_index_less_than_angle_center, int(len(number_index_less_than_angle_center)*shrink_rate) )
        new_index_left = np.random.choice(number_index_less_than_angle_left, int(len(number_index_less_than_angle_left)*shrink_rate) )
        new_index_right = np.random.choice(number_index_less_than_angle_right, int(len(number_index_less_than_angle_right)*shrink_rate) )
        
        print("randomly picked up center angle counts..",    len( new_index_center  ) )
        print("randomly picked up left angle counts..",    len( new_index_left  ) )
        print("randomly picked up right angle counts..",    len( new_index_right  ) )
        print("")
        # drop off  small angle data data
        Steering_center_small = Steering_center[new_index_center]  
        Steering_left_small = Steering_left[new_index_left] 
        Steering_right_small = Steering_right[new_index_right] 

        # concatenate steep steering angle + some straight line data
        Steering_center = np.concatenate( [ Steering_center[mask_center], Steering_center_small ] )
        Steering_left = np.concatenate( [ Steering_left[mask_left], Steering_left_small ] )
        Steering_right = np.concatenate( [Steering_right[mask_right], Steering_right_small ] )

        #
        # Images for small angle 
        #
        
        # new_index_XXX 
        centerImages_small = centerImages[new_index_center]
        leftImages_small = leftImages[new_index_left]
        rightImages_small = rightImages[new_index_right]

        centerImages = np.concatenate([ centerImages[mask_center], centerImages_small ])
        leftImages = np.concatenate([ leftImages[mask_left], leftImages_small ])
        rightImages = np.concatenate([ rightImages[mask_right], rightImages_small ])

        print("Integrated Images - center:%s  left:%s  right:%s " % (centerImages.shape, leftImages.shape, rightImages.shape  )   )
        print("Integrated Angles - center:%s  left:%s  right:%s " % (Steering_center.shape, Steering_left.shape, Steering_right.shape  )   )

        self.images = np.concatenate( [centerImages,leftImages,rightImages ], axis=0  )
        self.steerings = np.concatenate([Steering_center,Steering_left, Steering_right ],axis=0)

        #return images, steerings

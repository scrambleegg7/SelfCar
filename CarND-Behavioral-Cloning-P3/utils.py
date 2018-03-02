import numpy as np  
import pandas as pd 
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def displayImage(images, steerings):

    gs1 = gridspec.GridSpec(8, 4)
    gs1.update(wspace=0.01, hspace=0.1) # set the spacing between axes.
    plt.figure(figsize=(12,12))

    for i, (image, steering) in enumerate( zip( images, steerings )  ):        
        #print(name)

        ax1 = plt.subplot(gs1[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        #ax1.set_title( name )
        ax1.set_aspect('equal')
        #img = transform_image(image,20,10,5,brightness=1)

        plt.subplot(8,4,i+1)
        plt.title(steering)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


import numpy as np
import cv2
from skimage.io import imread

# Read in an image
image = imread('signs_vehicles_xygrad.png')

thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

plt.imshow(binary)
plt.show()
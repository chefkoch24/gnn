import numpy as np
import cv2 as cv2
from numpy import double

# Load an color image in grayscale
img = cv2.imread('lena.ppm')
img = double(img)/255
print(img)

filter = np.random.uniform(low=0, high=1, size=(3,3) )
print("filter",filter[0])
print("img",img[0][0 ])
sum = np.convolve(img[0][0], filter[0])
print("conv", sum)
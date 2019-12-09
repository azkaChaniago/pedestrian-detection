import cv2 as cv
import numpy as np

image = cv.imread('img/pah.png')

alpha = 3.0 # Simple contrast control
beta = 20   # Simple brightness control
new_image = np.zeros(image.shape, image.dtype)

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

cv.imwrite('img/pah_2.png', new_image)
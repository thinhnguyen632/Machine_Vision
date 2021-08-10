#Truong hop a

import cv2 as cv
import numpy as np

#Load image
img = cv.imread('8.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Binarize the image
b_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)

# Distance transform
dist_img = cv.distanceTransform(b_img, cv.DIST_L1, 3)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist_img, dist_img, 0, 1.0, cv.NORM_MINMAX)
out = cv.threshold(dist_img, 0.78, 1.0, cv.THRESH_BINARY)[1]

# Morphology
# create kernels
kernel_sq = np.array([[1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1]], dtype = np.uint8)
kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype = np.uint8)

out = cv.morphologyEx(out, cv.MORPH_ERODE, kernel_ci, iterations = 1)
out = cv.morphologyEx(out, cv.MORPH_DILATE, kernel_sq, iterations = 3)
out = cv.morphologyEx(out, cv.MORPH_ERODE, kernel_ci, iterations = 2)
out = cv.morphologyEx(out, cv.MORPH_DILATE, kernel_sq, iterations = 4)
out = cv.morphologyEx(out, cv.MORPH_ERODE, kernel_ci, iterations = 3)
out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel_sq, iterations = 6)

cv.imshow('result', out)
cv.waitKey(0)
cv.destroyAllWindows()
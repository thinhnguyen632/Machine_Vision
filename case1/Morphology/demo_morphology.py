import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img1 = cv.imread("Salt_noise.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("Small_holes.jpg", cv.IMREAD_GRAYSCALE)

# Convert to binary images
thresh = 120
b_img1 = cv.threshold(img1, thresh, 255, cv.THRESH_BINARY)[1]
b_img2 = cv.threshold(img2, thresh, 255, cv.THRESH_BINARY)[1]

roi = b_img2[:,0:149]

# Morphology
# create kernels
kernel_sq = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]], dtype = np.uint8)
# kernel_sq5 = np.array([[1,1,1,1,1],
#                        [1,1,1,1,1],
#                        [1,1,1,1,1],
#                        [1,1,1,1,1],
#                        [1,1,1,1,1]], dtype = np.uint8)
# kernel_sq7 = np.ones(49, dtype = np.uint8)
# kernel_sq7 = np.reshape(kernel_sq7, (7,7))
out1 = cv.morphologyEx(b_img1, cv.MORPH_OPEN, kernel_sq, iterations = 1)

kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype = np.uint8)
out2 = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel_ci, iterations = 8)

b_img2[:,0:149] = out2

cv.imshow("result1", out1)
cv.imshow("result2", b_img2)
cv.waitKey(0)
cv.destroyAllWindows()
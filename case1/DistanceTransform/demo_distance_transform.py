import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread("2cell.jpg", cv.IMREAD_GRAYSCALE)

# Binarize the image
b_img = cv.threshold(img, 120, 255, cv.THRESH_BINARY)[1]

# Distance transform
dist_img = cv.distanceTransform(b_img, cv.DIST_L2, 3)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist_img, dist_img, 0, 1.0, cv.NORM_MINMAX)
out = cv.threshold(dist_img, 0.8, 1.0, cv.THRESH_BINARY)[1]

# Show image
cv.imshow("input", b_img)
cv.imshow("distance", dist_img)
cv.imshow("output", out)
cv.waitKey(0)
cv.destroyAllWindows()
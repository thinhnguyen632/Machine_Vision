import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load anh color
img = cv.imread("cell1.jpg", cv.IMREAD_COLOR)

# Convert sang anh xam
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filter voi bo loc Gaussian/ Median
gray = cv.GaussianBlur(gray, (5,5), 0)

# Chuyen sang anh nhi phan
b_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 121, 2)

# Khu nhieu voi Morphology
kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype = np.uint8)
kernel2 = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype = np.uint8)
b_img = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel_ci, iterations = 2)
# b_img = cv.morphologyEx(b_img, cv.MORPH_CLOSE, kernel_ci, iterations = 2)
# b_img = cv.morphologyEx(b_img, cv.MORPH_ERODE, kernel2, interations = 2)

# Chuyen sang anh distance
dist_img = cv.distanceTransform(b_img, cv.DIST_L2, 3)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist_img, dist_img, 0, 1.0, cv.NORM_MINMAX)
out = cv.threshold(dist_img, 1, 1.0, cv.THRESH_BINARY)[1]
out = out.astype(np.uint8)

# find contour
contours, hierarchy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

# dem so contour, tao bounding box
# i = 1
# for c in contours:
#     area = cv.contourArea(c)
#     x,y,w,h = cv.boundingRect(c)
#     num = "#" + str(i)
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#     cv.putText(img.num,(x+w,y+h),cv.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
#     i = i + 1

# Show image
cv.imshow("b_img", b_img)
cv.imshow("result", out)
cv.waitKey(0)
cv.destroyAllWindows()
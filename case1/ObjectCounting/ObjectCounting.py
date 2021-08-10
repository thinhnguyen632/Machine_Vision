import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread("sample1.jpg", cv.IMREAD_COLOR) # BGR image

# Convert to GRAY image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot histogram
# hist = cv.calcHist([gray], [0], None, [256], [0,255])
# x = range(256)
# plt.plot(x,hist)
# plt.show()

# Segmentation
thresh = 210
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]

# Find contour
contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
print(hierarchy)

for c in contours:
    area = cv.contourArea(c)
    print(area)
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
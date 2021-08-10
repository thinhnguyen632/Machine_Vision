import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread("rice.jpg", cv.IMREAD_COLOR) # BGR image

# Convert to GRAY image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot histogram
#hist = cv.calcHist([gray], [0], None, [256], [0,255])
#x = range(256)
#plt.plot(x,hist)
#plt.show()

# Smooth filter
#gray1 = cv.blur(gray, [5,5])
gray = cv.medianBlur(gray, 5)
#gray3 = cv.GaussianBlur(gray, [5,5], 0)

# Segmentation
thresh = 130
#b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
b_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)

# Morphology to seperate rice seeds
kernel = np.ones((3,3),np.uint8)
b_img = cv.erode(b_img,kernel,iterations=2)

# Find contour
contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
#print(hierarchy)

for c in contours:
    area = cv.contourArea(c)
    #print(area)
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#cv.imshow("blur", gray1)
cv.imshow("median", b_img)
cv.imshow("result", img)
#cv.imshow("gaussian", gray3)
cv.waitKey(0)
cv.destroyAllWindows()
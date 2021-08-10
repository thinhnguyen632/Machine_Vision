import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('corners2.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # Harris Corner
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray, 2, 3, 0.04)
# #result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()

# Tomashi feature
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
cv.imshow("tomashi corners", img)
cv.waitKey(0)
cv.destroyAllWindows()
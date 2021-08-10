import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load file hinh
img = cv.imread("poly.png", cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# binarization
b_img = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]

# find contour
contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for c in contours:
    epsilon = 0.02*cv.arcLength(c,True)
    approx = cv.approxPolyDP(c,epsilon,True)
    if len(approx) == 3:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(img, " triangle", (x+w, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    if len(approx) == 4:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(img, " rectangle", (x+w, y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)



cv.imshow("raw image", img)
cv.waitKey(0)
cv.destroyAllWindows()
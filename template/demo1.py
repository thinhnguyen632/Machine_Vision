import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ham con xac dinh nguong
def determine_thres(gray):
    roi = gray[160:220, 120:200] # roi la numpy array 60x80
    thres = roi.mean() + 40
    return thres

img = cv.imread('rgb0002.jpg')

# chuyen sang anh xam
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# tim nguong threshold
thres = determine_thres(gray)

# nhi phan hoa hinh anh
b_img = cv.threshold(gray, thres, 255, cv.THRESH_BINARY)[1]

# ve do thi
data = gray[180,:] # data.shape =[320]
x = range(320) # x: 0 -> 319
plt.plot(x,data)
plt.show()

cv.imshow('image', img)
cv.imshow('binary', b_img)
cv.waitKey(0)
cv.destroyAllWindows()
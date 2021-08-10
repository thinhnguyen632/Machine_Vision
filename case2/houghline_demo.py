import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load file hinh
img = cv.imread("poly.png", cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# binarization
b_img = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
canny = cv.Canny(b_img, 150, 200,)
lines = cv.HoughLinesP(canny, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)# anh dau vao la anh nhi phan
# print(len(lines[0]))
for l in lines:
    x1,y1,x2,y2 = l[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow("canny", canny)
cv.imshow("result of Hough line transform", img)
cv.waitKey(0)
cv.destroyAllWindows()
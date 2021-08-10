import cv2 as cv
 
# Opens the Video file
cap= cv.VideoCapture('2balls.avi')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv.imwrite('C:/Users/Welcome/Documents/opencv/test3/frame/kang'+str(i)+'.jpg', frame)
    i+=1
 
cap.release()
cv.destroyAllWindows()
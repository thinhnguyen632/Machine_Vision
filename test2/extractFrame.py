import cv2 as cv
 
# Opens the Video file
cap= cv.VideoCapture('Flyingbird2.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv.imwrite('C:/Users/Welcome/Documents/opencv/test2/frame2/kang'+str(i)+'.jpg', frame)
    i+=1
 
cap.release()
cv.destroyAllWindows()
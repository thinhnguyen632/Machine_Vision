import numpy as np
import cv2 as cv
cap = cv.VideoCapture("6.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Your code
    # gaussian filter
    img = cv.GaussianBlur(img, (5,5), 0)

    # chuyen sang khong gian mau hsv
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)
    # h1 = cv.threshold(h,35,255,cv.THRESH_BINARY)[1]
    # h2 = cv.threshold(h,60,255,cv.THRESH_BINARY_INV)[1]
    # h = cv.bitwise_and(h1,h2)

    # dung ham inRange de tach color
    low_h = 20
    high_h= 65
    low_s = 50
    high_s= 250
    low_v = 50
    high_v= 250

    mask = cv.inRange(hsv,(low_h,low_s, low_v),(high_h,high_s,high_v))

    # morphology
    kernel_ci =  np.array([[0,0,1,0,0],
                        [0,1,1,1,0],
                        [1,1,1,1,1],
                        [0,1,1,1,0],
                        [0,0,1,0,0]], dtype=np.uint8)
    out = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_ci, iterations=1)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE,kernel_ci, iterations=3)

    # find countours
    contours, hierachy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # tim contour co bien dang hinh tron
    for c in contours:
        # tinh dien tich contour
        area = cv.contourArea(c)
        
        # size filter
        if (area<4000) and (area>120):

            # tim duong tron co ban kinh min -> tinh dien dien tich hinh tron nay
            (x,y),radius = cv.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)

            areaC = np.pi*(radius**2)
            # so sanh dien tich contour va dien tich hinh tron min -> dong khung
            if (1-area/areaC)<0.3:
                cv.circle(img,center,radius,(0,255,0),2)
                print("bounding box (x,y,w,h)=",center[0]-radius,center[1]-radius,2*radius, 2*radius )
        


    # show hinh
    cv.imshow("mask", out)
    cv.imshow("result", img)
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
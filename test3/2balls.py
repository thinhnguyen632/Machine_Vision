# Ho & Ten          MSSV
# Le Minh Thang     18146215
# Le Thanh Hai      18146108

import cv2 as cv
import numpy as np

# open video clip
cap = cv.VideoCapture('2balls.avi')
if not cap.isOpened():
    print("Cannot open camera")
    exit()



# Create tracker (KCF, TLD, MIL, CSRT)
tracker1 = cv.TrackerCSRT_create()
tracker2 = cv.TrackerCSRT_create()

def ball_detect(frame):
    # Convert to gray
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # dung ham inRange de tach color yellow
    low_h = 25
    high_h = 35
    low_s = low_v = 150
    high_s = high_v = 255
    mask = cv.inRange(hsv, np.array([low_h, low_s, low_v]), np.array([high_h, high_s, high_v]))

    # noise remove with morphology (optional)
    kernel_ci = np.array([[0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0]], dtype = np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_ci, iterations = 2)

    # find contours
    contours, hierachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # detect circle
    min_radius = 30
    max_radius = 60
    balls = np.empty((0, 4), dtype=np.uint8)
    for c in contours:
        (x,y),radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius) and (radius < max_radius):
            x, y, w, h = cv.boundingRect(c)
            balls = np.vstack((balls, np.array([x, y, w, h])))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv.imshow("Detecting . . . ",frame)
    return balls
    

    
x1_prev = None
dir_x1 = None
detect = True
while True:
    #cv.waitKey(150)
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if detect == True:
        balls = ball_detect(frame)
        if balls.shape[0] == 2:
            ret1 = tracker1.init(frame,balls[0])
            ret2 = tracker2.init(frame,balls[1])
            detect = False
    else:
        ret1, obj1 = tracker1.update(frame)
        ret2, obj2 = tracker2.update(frame)

        x1 , y1 , w1 , h1 = obj1
        x2 , y2 , w2 , h2 = obj2

        if ret1 or ret2:
            p3 = (int(x2),int(y2))
            p4 = (int(x2 + w2),int(y2 + h2))
            cv.rectangle(frame, p3,p4, (255,0,0), 2, 1)
            cv.putText(frame, "ball 2 ->", p3, cv.FONT_HERSHEY_SIMPLEX, 0.5, (230,14,201), 2)

            if x1_prev != None:
                dir_x1 = x1 - x1_prev
                if  (dir_x1 != None) and (dir_x1 < 0):
                    #print("Phai sang trai (dung huong)")
                    p1 = (int(x1),int(y1))
                    p2 = (int(x1 + w1),int(y1 + h1))
                    cv.rectangle(frame, p1,p2, (255,0,0), 2, 1)
                    cv.putText(frame, "<- ball 1", p1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,255), 2)
                else:
                    #print("Trai sang phai (sai huong)")
                    print("Detect again . . . ")
                    #detect = True

            # print('x1 :', x1)
            # print('x1_rev', x1_prev)
            # print('delta X1 ', dir_x1)
            
            x1_prev = x1

            # print('\n')

    cv.imshow("Tracking . . .",frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()


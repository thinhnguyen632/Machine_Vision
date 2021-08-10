#################################
# Ho va ten         # MSSV      #
# Nguyen Tien Thinh # 18146222  #
# Tran Quoc Huy     # 18146128  #
#################################

import cv2 as cv
import numpy as np
import time

# open video clip
cap = cv.VideoCapture('2balls.avi')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create tracker (KCF, TLD, MIL, CSRT)
tracker = cv.TrackerCSRT_create()
tracker1 = cv.TrackerCSRT_create()

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

    return balls

tracking = 0
crossed = 0

while True:
    # read next frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if tracking == 0:
        # detect ball
        balls = ball_detect(frame)

        # if 2 balls are detected -> tracking = 1
        if balls.shape[0] == 2:
            tracking = 1
            # initialize the tracker
            ret = tracker.init(frame, balls[0])
            ret1 = tracker1.init(frame, balls[1])

    else:
        ret, obj = tracker.update(frame)
        ret1, obj1 = tracker1.update(frame)

        if ret or ret1:
            p1 = (int(obj[0]), int(obj[1]))
            p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv.putText(frame, " #1", (obj[0] + obj[2], obj[1] + obj[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) 

            p3 = (int(obj1[0]), int(obj1[1]))
            p4 = (int(obj1[0] + obj1[2]), int(obj1[1] + obj1[3]))
            cv.rectangle(frame, p3, p4, (255,0,0), 2, 1)    
            cv.putText(frame, " #2", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)     
        
        # Lan dau 2 vat cham vao nhau => tracking = 0 (detect lai)
        if p1[0] - p3[0] < 0 and crossed == 0:
            tracking = 0
            crossed = 1
            print("detect again")
            
    cv.imshow("result", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
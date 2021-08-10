import cv2 as cv
import numpy as np

# open video clip
cap = cv.VideoCapture('2balls.avi')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracking = 0
tracker_init = 0

# load file cascade
balls_cascade = cv.CascadeClassifier()
balls_cascade.load("cascade.xml")

# Create tracker (KCF, TLD, MIL, CSRT)
tracker = cv.TrackerCSRT_create()
tracker1 = cv.TrackerCSRT_create()

# read each frame
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if tracking == 0:
        # Detect bang cascade
        balls = balls_cascade.detectMultiScale(frame)
        for (x, y, w, h) in balls:
            if len(balls) > 1 and (0 < w < 100) and (0 < h < 100):
                # initialize the tracker
                ret = tracker.init(frame, balls[0])
                ret1 = tracker1.init(frame, balls[1])

                tracking = 1
    
    else:
        balls = balls_cascade.detectMultiScale(frame)

        ret, obj = tracker.update(frame)
        ret1, obj1 = tracker1.update(frame)

        if len(balls) > 1 and (0 < w < 100) and (0 < h < 100):
            if ret:
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
                cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv.putText(frame, " #1", (obj[0] + obj[2], obj[1] + obj[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) 

            if ret1:
                p3 = (int(obj1[0]), int(obj1[1]))
                p4 = (int(obj1[0] + obj1[2]), int(obj1[1] + obj1[3]))
                cv.rectangle(frame, p3, p4, (255,0,0), 2, 1)
                cv.putText(frame, " #2", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        else:
            tracking = 0
            print("detect again")

            p1 = (int(obj[0]), int(obj[1]))
            p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv.putText(frame, " #1", (obj[0] + obj[2], obj[1] + obj[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            p3 = (int(obj1[0]), int(obj1[1]))
            p4 = (int(obj1[0] + obj1[2]), int(obj1[1] + obj1[3]))
            cv.rectangle(frame, p3, p4, (255,0,0), 2, 1)
            cv.putText(frame, " #2", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    cv.imshow("result", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



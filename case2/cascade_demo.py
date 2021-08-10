import numpy as np
import cv2 as cv
import time

# setup fps
frame_rate = 24
prev = 0

# load file cascade
ball_cascade = cv.CascadeClassifier()
ball_cascade.load("cascade.xml")
# load file / camera
cap = cv.VideoCapture("robocup.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    time_elapse = time.time() - prev
    if time_elapse > 1./frame_rate:
        prev = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Your code
        # Detect bang cascade
        balls = ball_cascade.detectMultiScale(frame)
        print(len(balls))
        for (x, y, w, h) in balls:
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
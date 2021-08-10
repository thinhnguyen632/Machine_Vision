# Nguyen Tien Thinh 18146222
# Tran Quoc Huy     18146128

import numpy as np
import cv2 as cv
import time

# setup fps
frame_rate = 24
prev = 0

# load file cascade
bird_cascade = cv.CascadeClassifier()
bird_cascade.load("bird2.xml")
# load file / camera
cap = cv.VideoCapture("Flyingbird2.mp4")
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
        birds = bird_cascade.detectMultiScale(frame)
        # print(len(birds))
        for (x, y, w, h) in birds:
            if (50 < w < 80 and 50 < h < 80):
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
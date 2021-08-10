import cv2 as cv
import numpy as np

cap = cv.VideoCapture('cr7.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create tracker (KCF, TLD, MIL, CSRT)
tracker = cv.TrackerCSRT_create()

# Load 1st frame
ret, frame = cap.read()
# frame = cv.resize(frame, (320, 340))

# detect object location
r = cv.selectROI(frame)
# Crop image
# roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# initialize the tracker
ret = tracker.init(frame, r)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.resize(frame, (320, 340))
    ret, obj = tracker.update(frame)
    if ret:
        p1 = (int(obj[0]), int(obj[1]))
        p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        tracking = 0
        print("tracking fail")

        # detect object

    cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
# import cv2 as cv
# import numpy as np

# cap = cv.VideoCapture("track1.mp4")
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# # Create some random colors
# color = np.random.randint(0,255,(100,3))

# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# # Load 1st frame
# ret, old_frame = cap.read()
# roi = old_frame[560:690,310:530]
# old_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
# # dectect object location

# # find keypoint on the Object
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# for i in p0:
    
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Your code
#     # Convert to gray
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]
#     # draw location
#     for i,(new,old) in enumerate(zip(good_new, good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
#         frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
#     img = cv.add(frame,mask)
#     # update current location
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
#     # ...
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
#%%
# import numpy as np
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('track1.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Create some random colors
color = np.random.randint(0,255,(100,3))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Load 1st frame
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# detect object location
r = cv.selectROI(old_frame)
# Crop image
roi = old_frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
# find keypoints on the Object
p0 = cv.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)
for i in p0:
    i[0,0] = i[0,0] + r[0]
    i[0,1] = i[0,1] + r[1]
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
#%%
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Convert to gray
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
     # draw location
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 8)
        frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    # update current location
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
    cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
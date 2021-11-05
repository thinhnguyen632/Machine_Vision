import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
from numpy.core.shape_base import _block_format_index
from numpy.lib.twodim_base import mask_indices

# add video
cap = cv.VideoCapture('vid_fix.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def ShapeDetect(frame):
    # gaussian filter
    frame = cv.GaussianBlur(frame, (5,5), 0)

    # print(frame.shape)
    # frame = cv.resize(frame, (1024, 576))

    # convert to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # binarization
    # b_frame = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)[1]
    b_frame = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 201, 81)
    roi = b_frame[100:550,420:604]

    # noise remove with morphology (optional)
    kernel_ci = np.array([[0,0,1,0,0],
                        [0,1,1,1,0],
                        [1,1,1,1,1],
                        [0,1,1,1,0],
                        [0,0,1,0,0]], dtype = np.uint8)
    kernel_ci_mini = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype = np.uint8)
    mask = cv.morphologyEx(roi, cv.MORPH_ERODE, kernel_ci_mini, iterations = 3)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel_ci, iterations = 4)

    # find contour
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    object = np.empty((0, 4), dtype=np.uint8)
    min_radius = 15
    max_radius = 20
    for c in contours:
        (x, y),radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        # print(radius)
        if (radius > min_radius) and (radius < max_radius):
            x, y, w, h = cv.boundingRect(c)
            # cv.rectangle(frame, (x + 420, y + 100), (x + w + 420, y + h + 100), (0,255,0), 2)
            object = np.vstack((object, np.array([x + 420, y + 100, w, h])))
    # cv.imshow("result", frame)
    # print(object.shape[0])
    return object

def UpDownDetect(frame, p1, p2, p3, p4):
    mlp = pickle.load(open('mlp_model.sav', 'rb'))

    # crop images
    d_size = (28, 28)
    crop1 = frame[p1[1]:p2[1],p1[0]:p2[0]]
    crop2 = frame[p3[1]:p4[1],p3[0]:p4[0]]
    crop1 = cv.resize(crop1, d_size)
    crop2 = cv.resize(crop2, d_size)

    # convert to gray
    gray1 = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY)
    gray1 = gray1.astype(np.float)
    gray2 = cv.cvtColor(crop2, cv.COLOR_BGR2GRAY)
    gray2 = gray2.astype(np.float)

    # preprocessing

    # normalize to range (0-1)
    cv.normalize(gray1, gray1, 0, 1.0, cv.NORM_MINMAX)
    cv.normalize(gray2, gray2, 0, 1.0, cv.NORM_MINMAX)

    # reshape to row vector
    gray1 = np.reshape(gray1, (1, 784))
    gray2 = np.reshape(gray2, (1, 784))

    result1= mlp.predict(gray1)
    if result1 == 0:
        text1 = ' [down]'
    else:
        text1 = ' [up]'
    result2 = mlp.predict(gray2)
    if result2 == 0:
        text2 = ' [down]'
    else:
        text2 = ' [up]'

    return text1, text2

def UpDownDetect_HSV(frame, p1, p2, p3, p4):
    mlp = pickle.load(open('mlp_model.sav', 'rb'))

    # crop images
    d_size = (28, 28)
    crop1 = frame[p1[1]:p2[1],p1[0]:p2[0]]
    crop2 = frame[p3[1]:p4[1],p3[0]:p4[0]]
    crop1 = cv.resize(crop1, d_size)
    crop2 = cv.resize(crop2, d_size)

    # convert to hsv
    hsv1 = cv.cvtColor(crop1, cv.COLOR_BGR2HSV)
    hsv2 = cv.cvtColor(crop2, cv.COLOR_BGR2HSV)
    
    # preprocessing
    low_H, high_H = (150, 170)
    low_S, high_S = (5, 255)
    low_V, high_V = (0, 250)
    kernel_ci = np.array([[0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0]], dtype = np.uint8)
    kernel_ci_mini = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype = np.uint8)
    
    hsv1 = cv.inRange(hsv1, (low_H, low_S, low_V), (high_H, high_S, high_V))
    hsv2 = cv.inRange(hsv2, (low_H, low_S, low_V), (high_H, high_S, high_V))

    hsv1 = cv.morphologyEx(hsv1, cv.MORPH_DILATE, kernel_ci_mini, iterations=1)
    hsv2 = cv.morphologyEx(hsv2, cv.MORPH_DILATE, kernel_ci_mini, iterations=1)

    hsv1 = hsv1.astype(np.float)
    hsv2 = hsv2.astype(np.float)

    # normalize to range (0-1)
    cv.normalize(hsv1, hsv1, 0, 1.0, cv.NORM_MINMAX)
    cv.normalize(hsv1, hsv1, 0, 1.0, cv.NORM_MINMAX)

    # reshape to row vector
    hsv1 = np.reshape(hsv1, (1, 784))
    hsv2 = np.reshape(hsv2, (1, 784))

    result1= mlp.predict(hsv1)
    if result1 == 0:
        text1 = ' [down]'
    else:
        text1 = ' [up]'
    result2 = mlp.predict(hsv2)
    if result2 == 0:
        text2 = ' [down]'
    else:
        text2 = ' [up]'

    return text1, text2

# Create tracker
tracker1 = cv.TrackerCSRT_create()
tracker2 = cv.TrackerCSRT_create()

tracking = 0
i = 0
j = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if tracking == 0:
        frame = cv.resize(frame, (1024, 576))
        # detect object
        object = ShapeDetect(frame)

        if object.shape[0] == 2:
            tracking = 1
            # initialize the tracker
            ret1 = tracker1.init(frame, object[0])
            ret2 = tracker2.init(frame, object[1])
    
    else:
        frame = cv.resize(frame, (1024, 576))

        # update tracker
        ret1, obj1 = tracker1.update(frame)
        ret2, obj2 = tracker2.update(frame)

        p1 = (int(obj1[0]), int(obj1[1]))
        p2 = (int(obj1[0] + obj1[2]), int(obj1[1] + obj1[3]))

        p3 = (int(obj2[0]), int(obj2[1]))
        p4 = (int(obj2[0] + obj2[2]), int(obj2[1] + obj2[3]))

        object = ShapeDetect(frame)

        # get data for training NN
        # if p1 != (0,0) and p2 != (0,0):
        #     crop1 = frame[p1[1]:p2[1],p1[0]:p2[0]]
        #     cv.imwrite('C:/Users/Welcome/Documents/opencv/FinalProject/cropped/kang1_'+str(i)+'.jpg', crop1)
        #     i = i + 1
        # if p3 != (0,0) and p4 != (0,0):
        #     crop2 = frame[p3[1]:p4[1],p3[0]:p4[0]]
        #     cv.imwrite('C:/Users/Welcome/Documents/opencv/FinalProject/cropped/kang2_'+str(j)+'.jpg', crop2)
        #     j = j +1

        if ret1 and ret2:
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv.rectangle(frame, p3, p4, (255,0,0), 2, 1) 

            # result = UpDownDetect(frame, p1, p2, p3, p4)
            result = UpDownDetect_HSV(frame, p1, p2, p3, p4)

            # id and detect up/ down object
            if p1[0] > 500:
                cv.putText(frame, " #1", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.putText(frame, result[0], (obj1[0] + obj1[2], obj1[1] + obj1[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv.putText(frame, " #2", (obj2[0] + obj2[2], obj2[1] + obj2[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.putText(frame, result[1], (obj2[0] + obj2[2], obj2[1] + obj2[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            else:
                cv.putText(frame, " #2", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.putText(frame, result[0], (obj1[0] + obj1[2], obj1[1] + obj1[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)  
                cv.putText(frame, " #1", (obj2[0] + obj2[2], obj2[1] + obj2[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv.putText(frame, result[1], (obj2[0] + obj2[2], obj2[1] + obj2[3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        if object.shape[0] != 2:
            tracking = 0
            print("detect again")

    cv.imshow("result", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
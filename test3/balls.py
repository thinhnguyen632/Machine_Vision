import cv2 as cv
import numpy as np


# open video clip
cap = cv.VideoCapture('2balls_25.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracking = 0

def ball_detect(frame):
    # convert to hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # yellow filter with inRange function
    low_H = 25
    high_H = 35
    low_S = low_V = 150
    high_S = high_V = 255
    mask = cv.inRange(hsv,np.array([low_H, low_S, low_V]),np.array([high_H,high_S,high_V]))

    # noise remove with morphology (optional)
    kernel_ci =  np.array([[0,0,1,0,0],
                        [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]], dtype=np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel_ci,iterations=2)
    # find contours
    contours, hierachy = cv.findContours(mask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # detect circle
    min_radius = 30
    max_radius = 60
    balls = np.empty((0, 4),dtype=np.uint8)
    for c in contours:
        (x,y),radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        # print(radius)
        if (radius > min_radius) and (radius < max_radius):
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            balls = np.vstack((balls,np.array([x,y,w,h])))
    cv.imshow("bbox",frame)
    # print(balls.shape)
    return balls

# read each frame
while True:
    # read next frame
    ret, frame = cap.read()
    if tracking == 0:
        # detect ball 
        balls = ball_detect(frame)
        # if 2 balls are detected -> tracking = 1
        print(balls.shape)
        if balls.shape[0] == 2:
            tracking = 1

            print(balls[0,:])
            print(balls[1,:])

            ball_1= balls[0,:]
            ball_2= balls[1,:]
            
    else:
        pass
        # get balls ROIs

        # get keypoints of each ROIs
        
        # use KLT algorithm to predict balls' location  
        
        # Draw it on image    
        tracker = cv.TrackerCSRT_create()
        tracker2 = cv.TrackerCSRT_create()
        r1 = ball_1
        r2 = ball_2
        ret1 = tracker.init(frame, r1)
        ret2 = tracker2.init(frame, r2)



        ##############################

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # frame = cv.resize(frame,(320,240))
            ret1, obj1 = tracker.update(frame)
            ret2, obj2 = tracker2.update(frame)
            if ret1 and ret2:
                p1 = (int(obj1[0]),int(obj1[1]))
                p2 = (int(obj1[0] + obj1[2]),int(obj1[1] + obj1[3]))
                cv.rectangle(frame, p1,p2, (255,0,0), 2, 1)
                cv.putText(frame, " ball_1", (obj1[0] + obj1[2], obj1[1] + obj1[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                p3 = (int(obj2[0]),int(obj2[1]))
                p4 = (int(obj2[0] + obj2[2]),int(obj2[1] + obj2[3]))
                cv.rectangle(frame, p3,p4, (255,0,0), 2, 1)
                cv.putText(frame, " ball_2", (obj2[0] + obj2[2], obj2[1] + obj2[3]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                print(' ')
                print(p1,p2)
                print(p3,p4)
                
            
            # elif ret2 :
            #     p3 = (int(obj2[0]),int(obj2[1]))
            #     p4 = (int(obj2[0] + obj2[2]),int(obj2[1] + obj2[3]))

            #     cv.rectangle(frame, p3,p4, (255,0,0), 2, 1)


            else:
                print("tracking fail")
                # detect object

            cv.imshow("first frame", frame)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break




    # cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
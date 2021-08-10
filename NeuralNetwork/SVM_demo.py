# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Path of image folder
from pathlib import Path
img_folder = Path('/home/tof/Documents/opencv/GenderDB')
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'
test_images = list(test_folder.glob('*.png'))
train_images = list(train_folder.glob('*.png'))

# %%

# def on_trackbar(val):
#     image_path = str(train_images[val])
#     img = cv.imread(str(image_path),cv.IMREAD_COLOR)
#     cv.imshow('Images', img)
    

# num_images = len(train_images)

# cv.namedWindow('Images')
# trackbar_name = 'Show image: '
# cv.createTrackbar(trackbar_name, 'Images', 0, num_images-1, on_trackbar)

# cv.waitKey(0)
# cv.destroyAllWindows()

# %%
# Create training data and target 
num_images = len(train_images)
train_data = np.empty((0, 2700),dtype=np.float)
train_target = np.empty(num_images,dtype=np.uint8)

for i in range(num_images):
    # train_data
    image_path = str(train_images[i])
    img = cv.imread(str(image_path),cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.equalizeHist(gray)
    gray = gray.astype(np.float)
    # normalize to range [0,1]
    cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)
    # convert to column vector
    gray = np.reshape(gray,(1,2700))
    train_data = np.vstack((train_data,gray))
    # target
    if train_images[i].stem[0] == 'f':
        train_target[i] = 0
    else:
        train_target[i] = 1
# print(train_data.shape)
# print(target)
# Create test data and target 
num_images = len(test_images)
test_data = np.empty((0, 2700),dtype=np.float)
test_target = np.zeros(num_images,dtype=np.uint8)
for i in range(num_images):
    # test_data
    image_path = str(test_images[i])
    img = cv.imread(str(image_path),cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.equalizeHist(gray)
    gray = gray.astype(np.float)
    # normalize to range [0,1]
    cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)
    # convert to column vector
    gray = np.reshape(gray,(1,2700))
    test_data = np.vstack((test_data,gray))
    # result 
    if test_images[i].stem[0] == 'f':
        test_target[i] = 0
    else:
        test_target[i] = 1

#%% Build and train SVM model
from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(train_data, train_target)

# %%
from sklearn.metrics import classification_report,confusion_matrix

predict_train = clf.predict(train_data)
predict_test = clf.predict(test_data)
# print(confusion_matrix(train_target,predict_train))
# print(classification_report(train_target,predict_train))

print(confusion_matrix(test_target,predict_test))
print(classification_report(test_target,predict_test))

# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# %%
img_folder = Path('C:/Users/Welcome/Documents/opencv/NN/DogvsCat')
test_folder = img_folder / 'test1'
train_folder = img_folder / 'train'
test_images = list(test_folder.glob('*.jpg'))
train_images = list(train_folder.glob('*.jpg'))

# %% Prepare training and test data
# training data and target
d_size = (100, 100)
num_images = len(train_images)
train_data = np.empty((0,10000), dtype=np.float)
train_target = np.empty(num_images, dtype=np.int)

for i in np.arange(0, num_images, 1):
    print("Load images...")
    # load image
    img_path = str(train_images[i])
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, d_size)
    # convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float)
    # preprocessing

    # normalize to range (0-1)
    cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)
    # reshape to row vector
    gray = np.reshape(gray, (1,10000))
    # stack to train_data
    train_data = np.vstack((train_data, gray))
    # read filename -> target value
    if train_images[i].stem[0:2] == 'cat':
        train_target[i] = 0
    else:
        train_target[i] = 1
        
print(train_data.shape)
print(train_target.shape)

#!/usr/bin/env python
# coding: utf-8

# In[23]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# In[24]:


# create list of images
img_folder = Path('C:/Users/Welcome/Documents/opencv/FinalProject/data')
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'
test_images = list(test_folder.glob('*.jpg'))
train_images = list(train_folder.glob('*.jpg'))


# In[25]:


low_H, high_H = (150, 170) #170 - 200
low_S, high_S = (5, 255)
low_V, high_V = (0, 250)
d_size = (28, 28)
kernel_ci = np.array([[0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0]], dtype = np.uint8)
kernel_ci_mini = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype = np.uint8)


# In[26]:


# Prepare training and test data
# training data and target
num_images = len(train_images)
train_data = np.empty((0,784), dtype=np.float)
train_target = np.empty(num_images, dtype=np.int)


# In[27]:


for i in range(num_images):
    # load image
    img_path = str(train_images[i])
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, d_size)
    
    # convert to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # preprocessing
    hsv = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    hsv = cv.morphologyEx(hsv, cv.MORPH_DILATE, kernel_ci, iterations=1)
    hsv = hsv.astype(np.float)

    # normalize to range (0-1)
    cv.normalize(hsv, hsv, 0, 1.0, cv.NORM_MINMAX)
    
    # reshape to row vector
    hsv = np.reshape(hsv, (1, 784))
    
    # stack to train_data
    train_data = np.vstack((train_data, hsv))
    
    # read filename -> target value
    if train_images[i].stem[0] == 'd':
        train_target[i] = 0
    else:
        train_target[i] = 1


# In[28]:


# test data and target
num_images = len(test_images)
test_data = np.empty((0,784), dtype=np.float)
test_target = np.empty(num_images, dtype=np.int)


# In[29]:


for i in range(num_images):
    # load image
    img_path = str(test_images[i])
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = cv.resize(img, d_size)
    
    # convert to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # preprocessing
    hsv = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    hsv = cv.morphologyEx(hsv, cv.MORPH_DILATE, kernel_ci, iterations=1)
    hsv = hsv.astype(np.float)
    
    # normalize to range (0-1)
    cv.normalize(hsv, hsv, 0, 1.0, cv.NORM_MINMAX)
    
    # reshape to row vector
    hsv = np.reshape(hsv, (1, 784))
    
    # stack to train_data
    test_data = np.vstack((test_data, hsv))
    
    # read filename -> target value
    if test_images[i].stem[0] == 'd':
        test_target[i] = 0
    else:
        test_target[i] = 1


# In[30]:


# Build Neural Network
# 2 hidden layer, 15 nerons each layer
mlp = MLPClassifier(hidden_layer_sizes = (32, 16),
                    activation = 'relu', 
                    solver = 'adam',
                    batch_size = 20,
                    max_iter = 500)


# In[31]:


# train with training data
mlp.fit(train_data, train_target)


# In[32]:


# test and print result
result_train = mlp.predict(train_data)
result_test = mlp.predict(test_data)
print(confusion_matrix(test_target, result_test))
print(classification_report(test_target, result_test))


# In[33]:


import pickle
filename = "C:/Users/Welcome/Documents/opencv/FinalProject/data/mlp_model.sav"
pickle.dump(mlp, open(filename, 'wb'))


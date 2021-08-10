# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# create list of images
from pathlib import Path

img_folder = Path('C:/Users/Welcome/Documents/opencv/NN/GenderDB')
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'
test_images = list(test_folder.glob('*.png'))
train_images = list(train_folder.glob('*.png'))
# %% Prepare training and test data
# training data and target
num_images = len(train_images)
train_data = np.empty((0,2700), dtype=np.float)
train_target = np.empty(num_images, dtype=np.int)

for i in range(num_images):
    # load image
    img_path = str(train_images[i])
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    # convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float)
    # preprocessing

    # normalize to range (0-1)
    cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)
    # reshape to row vector
    gray = np.reshape(gray, (1, 2700))
    # stack to train_data
    train_data = np.vstack((train_data, gray))
    # read filename -> target value
    if train_images[i].stem[0] == 'f':
        train_target[i] = 0
    else:
        train_target[i] = 1
# print(train_data.shape)
# print(train_target.shape)

# test data and target
num_images = len(test_images)
test_data = np.empty((0,2700), dtype=np.float)
test_target = np.empty(num_images, dtype=np.int)

for i in range(num_images):
    # load image
    img_path = str(test_images[i])
    img = cv.imread(img_path, cv.IMREAD_COLOR )
    # convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float)
    # preprocessing

    # normalize to range (0-1)
    cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)
    # reshape to row vector
    gray = np.reshape(gray, (1, 2700))
    # stack to train_data
    test_data = np.vstack((test_data, gray))
    # read filename -> target value
    if test_images[i].stem[0] == 'f':
        test_target[i] = 0
    else:
        test_target[i] = 1
# print(test_data.shape)
# print(test_target.shape)


# %% Build Neural Network
# 2 hidden layer, 15 neurons each layer
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes = (15, 15),
                    activation = 'relu', 
                    solver = 'adam',
                    batch_size = 20,
                    max_iter = 500)
# train with training data
mlp.fit(train_data, train_target)

#%% test and print result
from sklearn.metrics import classification_report, confusion_matrix

result_train = mlp.predict(train_data)
result_test = mlp.predict(test_data)

# print(confusion_matrix(train_target, result_train))
# print(classification_report(train_target, result_train))

print(confusion_matrix(test_target, result_test))
print(classification_report(test_target, result_test))


# %% save model for future work
import pickle
filename = "mlp_model.sav"
pickle.dump(mlp, open(filename, 'wb'))
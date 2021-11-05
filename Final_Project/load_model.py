import cv2 as cv
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

mlp = pickle.load(open('mlp_model.sav', 'rb'))

# load images
d_size = (28, 28)
img = cv.imread('kang1_218.jpg', cv.IMREAD_COLOR)
img_resized = cv.resize(img, d_size)

# convert to gray
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
gray = gray.astype(np.float)

# preprocessing

# normalize to range (0-1)
cv.normalize(gray, gray, 0, 1.0, cv.NORM_MINMAX)

# reshape to row vector
gray = np.reshape(gray, (1, 784))

result = mlp.predict(gray)
print(result)

cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()
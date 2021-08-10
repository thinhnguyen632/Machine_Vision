import cv2 as cv
import numpy as np
import random
from os import listdir

# Khai bao thu muc chua cac frame va thu muc save frame sau khi crop
path = "C:/Users/Welcome/Documents/opencv/test3/frame"
save_path = "C:/Users/Welcome/Documents/opencv/test3/n"

# Kich thuoc crop nho nhat
min_h = 256
min_w = 256

i=0
for file in listdir(path):
    if file!='.DS_Store':
        print("File=", file)
        # Doc hinh anh
        img = cv.imread(path + "/" + file, cv.IMREAD_COLOR)

        # Cat hinh anh
        height = img.shape[0]
        width = img.shape[1]
        x = random.randint(0, width - min_w)
        y = random.randint(0, height - min_h)
        h = random.randint(min_h, height)
        w = random.randint(min_w, width)
        crop_img = img[y:y+h, x:x+w]

        # Luu hinh anh
        cv.imwrite(save_path + '/cropped_10_' + str(i) + '.jpg', crop_img)
        i = i + 1
        

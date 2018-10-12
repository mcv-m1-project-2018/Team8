import cv2
import os
from main import CONSOLE_ARGUMENTS
import numpy as np

directory = CONSOLE_ARGUMENTS.im_directory
dirs =os.listdir(directory)
square_kernel = np.ones((3,3), np.uint8)

for im_path in dirs:
    img = cv2.imread(directory+'/'+im_path)
    # Opening
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, square_kernel)
    cv2.imshow('mod',opening*255)
    cv2.imshow('original', img*255)
    cv2.waitKey()
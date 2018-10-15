import cv2 as cv
import os
from PIL import Image, ImageDraw
from main import CONSOLE_ARGUMENTS
import matplotlib.pyplot as plt
import numpy as np
from candidate_generation_pixel import morf_method1, boundingBoxFilter_method1,detectBoundingBoxes



def boundingBoxFilter(bb_list, im):
    image = im.copy()
    pixels = []
    for x,y,w,h in bb_list:
        pixels.append(w*h)
    mean = np.mean(pixels)
    desvest = np.std(pixels)

    print(mean, desvest)
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 700 or w*h > 75000 or f_ratio < 0.3 or form_factor < 0.333 or form_factor > 3):
            image[y:y+h, x:x+w] = np.zeros((h,w))

    return image


    

directory = CONSOLE_ARGUMENTS.im_directory
dirs = os.listdir(directory)
big_kernel = np.ones((6,6), np.uint8)
biger_kernel = np.ones((8,8), np.uint8)
bigerer_kernel = np.ones((14,14), np.uint8)
bigerest_kernel = np.ones((20,20), np.uint8)

norm_kernel = np.ones((4,4), np.uint8)
small_kernel = np.ones((3,3), np.uint8)
smaller_kernel = np.ones((2,2), np.uint8)
# smallest_kernel = np.ones((1,1), np.uint8)
h_kernel = np.ones((1,4), np.uint8)
v_kernel = np.ones((4,1), np.uint8)

# def fill_holes(im_th):
#     im2, contours, hierarchy = cv.findContours(im_th,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         cv.drawContours(im_th,[cnt],0,255,-1)

#     return im_th

for im_path in dirs:
    img = cv.imread(directory+'/'+im_path)
    imggray = cv.cvtColor(img.copy(), cv.COLOR_RGB2GRAY)
    base, extension = os.path.splitext(im_path)
    imorg = cv.imread("./Dataset/train/"+base+".jpg")
    # Opening
    opening = imggray.copy()
    imagen = morf_method1(opening)

    bb_list = detectBoundingBoxes(imagen)
    imagen = boundingBoxFilter_method1(imagen)

#     for x,y,w,h in bb_list:
#         cv.rectangle(imagen,(x,y),(x+w,y+h),(200,0,0),2)
    masked = imorg*(np.dstack([imagen]*3))

    small_im = cv.resize(img, (0,0), fx=0.5, fy=0.5)
    small_op = cv.resize(opening, (0,0), fx=0.5, fy=0.5) 
    small_fill = cv.resize(imagen, (0,0), fx=0.5, fy=0.5) 
    small_masked = cv.resize(masked, (0,0), fx=0.5, fy=0.5) 


    cv.imshow('fill',small_fill)
    cv.imshow('original', small_im)
    cv.imshow('masked', small_masked)
    k = cv.waitKey()
    if k==27: # Esc key to stop
        break
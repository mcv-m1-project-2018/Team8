import cv2
import os
from PIL import Image, ImageDraw
from main import CONSOLE_ARGUMENTS
import matplotlib.pyplot as plt
import numpy as np


def fill_holes(im):
    im_th = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)

    _, contours, _ = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(im_th,[cnt],0,255,-1)
    
    return im_th

def detectBoundingBoxes(im):
    _, contours, _ = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bb_list = list()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)   
        bb_list.append((x,y,w,h))


    return bb_list

def boundingBoxFilter(bb_list, im):
    image = im.copy()
    for x,y,w,h in bb_list:
        if(w < 25 and h < 25):
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

def fill_holes(im_th):
    im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(im_th,[cnt],0,255,-1)

    return im_th

for im_path in dirs:
    img = cv2.imread(directory+'/'+im_path)
    # Opening
    imggray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    opening = imggray.copy()
    opening = cv2.erode(opening,smaller_kernel,iterations=1)
#     opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, small_kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, small_kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, h_kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, v_kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, bigerer_kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, biger_kernel)
    # opening = cv2.erode(opening,bigerer_kernel,iterations=1)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, biger_kernel)
    # opening = cv2.dilate(opening,bigerest_kernel,iterations=1)
    imagen = fill_holes(opening)

    bb_list = detectBoundingBoxes(imagen)
    imagen = boundingBoxFilter(bb_list,imagen)

    small_op = cv2.resize(opening, (0,0), fx=0.5, fy=0.5) 
    small_fill = cv2.resize(imagen, (0,0), fx=0.5, fy=0.5) 
    small_im = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    cv2.imshow('mod',small_op*255)
    cv2.imshow('fill',small_fill)
    cv2.imshow('original', small_im*255)
    k = cv2.waitKey()
    if k==27: # Esc key to stop
        break
import cv2
import os
from PIL import Image, ImageDraw
from main import CONSOLE_ARGUMENTS
import matplotlib.pyplot as plt
import numpy as np
from candidate_generation_pixel import morf_method1, boundingBoxFilter_method1

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
#     im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         cv2.drawContours(im_th,[cnt],0,255,-1)

#     return im_th

for im_path in dirs:
    img = cv2.imread(directory+'/'+im_path)
    base, extension = os.path.splitext(im_path)
    imorg = cv2.imread("./Dataset/train/"+base+".jpg")
    # Opening
    opening = img.copy()
    imagen = morf_method1(opening)

    bb_list = detectBoundingBoxes(imagen)
    imagen = boundingBoxFilter_method1(bb_list,imagen)
#     for x,y,w,h in bb_list:
#         cv2.rectangle(imagen,(x,y),(x+w,y+h),(200,0,0),2)
    masked = imorg*(np.dstack([imagen]*3))

    small_im = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    small_op = cv2.resize(opening, (0,0), fx=0.5, fy=0.5) 
    small_fill = cv2.resize(imagen, (0,0), fx=0.5, fy=0.5) 
    small_masked = cv2.resize(masked, (0,0), fx=0.5, fy=0.5) 


    cv2.imshow('fill',small_fill)
    cv2.imshow('original', small_im*255)
    cv2.imshow('masked', small_masked)
    k = cv2.waitKey()
    if k==27: # Esc key to stop
        break
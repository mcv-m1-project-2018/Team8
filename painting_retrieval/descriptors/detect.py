# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 18:48:18 2018

@author: all
"""
import cv2 as cv
from descriptors.utils import detector_s
import os

from tqdm import tqdm

import pickle as pckl
import imutils
import cv2

from preprocess.detect_textbox import generateMaskFrombb
from preprocess.utils import rotate_points, resize_keeping_ar
kp_folder_name = "keypoints/"


def save_kp_img(filename, kp_list, factor):
    index = []
    index.append(factor)
    for point in kp_list[1:]:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
        index.append(temp)
    
    # Dump the keypoints
    f = open(filename, "wb")
    pckl.dump(index,f)
    f.close()

def load_kp_img(filename):
    with open(filename, 'rb') as pickle_file:
        index = pckl.load(pickle_file)

    kp_list = []
    factor = index[0]
    for point in index[1:]:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
                                _response=point[3], _octave=point[4], _class_id=point[5]) 
        kp_list.append(temp)
    return kp_list, factor


def detect_kp(img, detector, colorspace="gray", mask=None):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    detgen = detector_s[detector](1000)
    kp = detgen.detect(img,mask)
    return kp

def detect_all_kp(names, path, descriptor, colorspace="gray", image_width=-1,\
                  mask=[], rot_rectangle = [], crop=True):
    
    kpfolder = path+kp_folder_name
    if not os.path.exists(kpfolder):
        os.makedirs(kpfolder)
        
#    desc_all = []
    kp_all = []
    factors = []
    for i, name in tqdm(enumerate(names), desc="Detecting KeyPoints"):
        filename = kpfolder+descriptor+"_"+str(image_width)+"_"+name+".txt"
        if(os.path.isfile(filename)):
            kp_list, factor = load_kp_img(filename)
        else:
            img = cv.imread(path+name)
            factor = 1
            if(image_width > 0):
                img, factor = resize_keeping_ar(img, image_width)
            if(len(rot_rectangle) > 0):
                ang = rot_rectangle[i][0]
                old_h,old_w = img.shape[:2]
#                cv2.imshow("original img", img)
#                img_orig = img.copy()
                img = imutils.rotate_bound(img, ang+180)
#                cv2.imshow("rotated img", img)
                if(crop):
                    rot_h,rot_w = img.shape[:2]
                    old_c = (old_w/2, old_h/2)
                    rot_c = (rot_w/2, rot_h/2)
                    dif_c = (rot_c[0]-old_c[0],rot_c[1]-old_c[1])
                    print(dif_c)
                    pts = rot_rectangle[i][1]
                    
                    ####### PRINT OVER UNROTATED
#                    img2 = img_orig.copy()
#                    radius = int(max(old_w/100, 5))
#                    thick = int(max(old_w/60-1, 5-1))
#                    color = (0,0,255)
#                    
#                    cv.circle(img2, pts[0], radius, color, thickness=thick)
#                    cv.circle(img2, pts[1], radius, color, thickness=thick)
#                    cv.circle(img2, pts[2], radius, color, thickness=thick)
#                    cv.circle(img2, pts[3], radius, color, thickness=thick)
#                    cv.imshow('Unrotated rectangle',resize_keeping_ar(img2)[0])
                    ############################
                    inc_point = lambda p: (int(p[0]+dif_c[0]), int(p[1]+dif_c[1]))
                    rpts = rotate_points(pts, ang+180, old_c)
                    rpts = [inc_point(p) for p in rpts]
                    maxX = max([p[0] for p in rpts])
                    minX = min([p[0] for p in rpts])
                    maxY = max([p[1] for p in rpts])
                    minY = min([p[1] for p in rpts])
#                    bX = int(minX+dif_c[0])
#                    tX = int(maxX+dif_c[0])
#                    bY = int(minY+dif_c[1])
#                    tY = int(maxY+dif_c[1])
                    
                    ####### PRINT OVER ROTATED
#                    img3 = img.copy()
#                    cv.circle(img3, rpts[0], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[1], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[2], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[3], radius, color, thickness=thick)
#                    cv.imshow('rotated rectangle',resize_keeping_ar(img3)[0])
                    ##########################
                    img = img[minY:maxY, minX:maxX]
#                    cv2.imwrite("crop"+str(i)+".png",img)
#                    cv2.imshow("cropped img", resize_keeping_ar(img)[0])
                    cv2.waitKey()
            if(len(mask) > 0):
#                m = mask[i]
                mbox = mask[i]
                m = generateMaskFrombb(mbox, img.shape)
                if(image_width > 0):
                    m = resize_keeping_ar(mask[i], image_width)

                kp_list = detect_kp(img, descriptor, colorspace=colorspace, mask=m.astype('uint8'))
            else:
                kp_list = detect_kp(img, descriptor, colorspace=colorspace, mask=None)
            save_kp_img(filename, kp_list, factor)
#        desc_all.append(desc)
        kp_all.append(kp_list)
        factors.append(factor)
    return kp_all, factors
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

kp_folder_name = "keypoints/"

def resize_keeping_ar(im, desired_width=300):
    height, width, __ = im.shape
    factor = width/float(desired_width)
    desired_height = int(height/factor)
    imres = cv2.resize(im, (desired_width, desired_height))
    return imres, factor

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

def detect_all_kp(names, path, descriptor, colorspace="gray", image_width=-1, mask=[], rotAngle = []):
    
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
            if(len(rotAngle) > 0):
                img = imutils.rotate_bound(img, rotAngle[i])
            
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
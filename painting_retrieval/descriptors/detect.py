# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 18:48:18 2018

@author: hamdd
"""
import cv2 as cv
from descriptors.utils import detector_s
import os

from tqdm import tqdm

import pickle as pckl
import cv2

kp_folder_name = "keypoints/"
def save_kp_img(filename, kp_list):
    index = []
    for point in kp_list:
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
    
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
                                _response=point[3], _octave=point[4], _class_id=point[5]) 
        kp_list.append(temp)
    return kp_list


def detect_kp(img, detector, colorspace="gray"):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    detgen = detector_s[detector]()
    kp = detgen.detect(img,None)
    return kp

def detect_all_kp(names, path, descriptor, colorspace="gray"):
    kpfolder = path+kp_folder_name
    if not os.path.exists(kpfolder):
        os.makedirs(kpfolder)
        
#    desc_all = []
    kp_all = []
    for i, name in tqdm(enumerate(names), desc="Detecting KeyPoints"):
        filename = kpfolder+descriptor+"_"+name+".txt"
        if(os.path.isfile(filename)):
            kp_list = load_kp_img(filename)
        else:
            img = cv.imread(path+name)
            kp_list = detect_kp(img, descriptor, colorspace=colorspace)
            save_kp_img(filename, kp_list)
#        desc_all.append(desc)
        kp_all.append(kp_list)
    return kp_all
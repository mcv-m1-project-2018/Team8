# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 18:48:18 2018

@author: hamdd
"""
import cv2 as cv
from descriptors.utils import detector_s

from tqdm import tqdm

def detect_kp(img, detector, colorspace="gray"):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    detgen = detector_s[detector]()
    kp = detgen.detect(img,None)
    return kp

def detect_all_kp(names, path, descriptor, colorspace="gray"):
    desc_all = []
    kp_all = []
    for i, name in tqdm(enumerate(names), desc="Detecting KeyPoints"):
        img = cv.imread(path+name)
        kp = detect_kp(img, descriptor, colorspace=colorspace)
#        desc_all.append(desc)
        kp_all.append(kp)
    return kp_all
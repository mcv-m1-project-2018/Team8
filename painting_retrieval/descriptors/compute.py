# -*- coding: utf-8 -*-
import cv2 as cv
from tqdm import tqdm
from descriptors.utils import compute_s

def compute_features(img, kp, descriptor, colorspace="gray"):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    descgen = compute_s[descriptor]()
    kp,desc = descgen.compute(img, kp)
    return kp, desc

def compute_all_features(names, path, kp_list,  descriptor, colorspace="gray"):
    desc_all = []
    kp_all = []
    for i, name in tqdm(enumerate(names), desc="Computing descriptors"):
        img = cv.imread(path+name)
        kp, desc = compute_features(img, kp_list[0], descriptor, colorspace=colorspace)
        desc_all.append(desc)
        kp_all.append(kp)
    return kp_all, desc_all
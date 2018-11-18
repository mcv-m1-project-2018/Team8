# -*- coding: utf-8 -*-
import cv2 as cv
from tqdm import tqdm
from descriptors.utils import compute_s

import pickle as pckl

import numpy as np
import os
import imutils

from preprocess.utils import rotate_and_crop

desc_folder_name = "descriptors/"
def save_desc_img(filename, desc_array):
    # Dump the keypoints
    np.save(filename, desc_array)
#    pckl.dump(index,f)
#    f.close()

def load_desc_img(filename):
    desc_array = np.load(filename)
    return desc_array

def compute_features(img, kp, descriptor, colorspace="gray"):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    descgen = compute_s[descriptor]()
    kp,desc = descgen.compute(img, kp)
    return kp, desc


def compute_all_features(names, path, kp_list,  descriptor, detector, \
                         colorspace="gray", image_width=-1, \
                         rot_rectangle = [], crop=True):
    descfolder = path+desc_folder_name
    if not os.path.exists(descfolder):
        os.makedirs(descfolder)
        
    desc_all = []
    kp_all = []
    for i, name in tqdm(enumerate(names), desc="Computing descriptors"):
        filename = descfolder+detector+descriptor+"_"+str(image_width)+"_"+name
        if(os.path.isfile(filename+".npy")):
            desc = load_desc_img(filename+".npy")
        else:
            img = cv.imread(path+name)
            if(len(rot_rectangle) > 0):
                ang = rot_rectangle[i][0]
                if(crop):
                    img = rotate_and_crop(img, ang, rot_rectangle[i][1])
                else:
                    img = imutils.rotate_bound(img, ang+180)
            kp, desc = compute_features(img, kp_list[i], descriptor, colorspace=colorspace)
            save_desc_img(filename, desc)
        desc_all.append(desc)
        kp_all.append(kp_list[i])
    return kp_all, desc_all
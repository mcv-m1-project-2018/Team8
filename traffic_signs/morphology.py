# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def fill_holes(im):
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(im,[cnt],0,255,-1)
    return im

def morf_method1(im):
    imggray = im.copy()
    # print(im.shape)
    # imggray = cv.cvtColor(im.copy(), cv.COLOR_RGB2GRAY)
    opening = imggray.copy()
    opening = cv.erode(opening,(2,2),iterations=1)
#     opening = cv.morphologyEx(opening, cv.MORPH_OPEN, small_kernel)
    # big_kernel = np.ones((6,6), np.uint8)
    biger_kernel = np.ones((8,8), np.uint8)
    bigerer_kernel = np.ones((14,14), np.uint8)
    bigerest_kernel = np.ones((20,20), np.uint8)

    # norm_kernel = np.ones((4,4), np.uint8)
    small_kernel = np.ones((3,3), np.uint8)
    # smaller_kernel = np.ones((2,2), np.uint8)
    # smallest_kernel = np.ones((1,1), np.uint8)
    h_kernel = np.ones((1,4), np.uint8)
    v_kernel = np.ones((4,1), np.uint8)

    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, small_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, h_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, v_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerer_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, biger_kernel)
    # opening = cv.erode(opening,bigerer_kernel,iterations=1)
    # opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    # opening = cv.dilate(opening,bigerest_kernel,iterations=1)
    imagen = fill_holes(opening)

    rows,cols= imagen.shape
    M = np.float32([[1,0,-5],[0,1,-5]])
    imagen = cv.warpAffine(imagen,M,(cols,rows))

    return imagen

def apply_morphology(im, morphology):
    switcher_morf = {
        'm1': morf_method1
    }
    
    # PIXEL MORPHOLOGY
    if morphology is not None:
        if not isinstance(morphology, list):
            morphology = list(morphology)
        for preproc in morphology:
            func = switcher_morf.get(preproc, lambda: "Invalid morphology")
            im = func(im)
    return im
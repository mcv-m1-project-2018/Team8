# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def fill_holes(im):
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(im,[cnt],0,255,-1)
    return im

small_kernel = np.ones((3,3), np.uint8)
medium_kernel = np.ones((8,8), np.uint8)
big_kernel = np.ones((14,14), np.uint8)
huge_kernel = np.ones((20,20), np.uint8)

h_kernel = np.ones((1,4), np.uint8)
v_kernel = np.ones((4,1), np.uint8)
    
def morph_method1(im):
    imggray = im.copy()

    morph = imggray.copy()
    morph = cv.erode(morph,(2,2),iterations=1)
  
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, huge_kernel)
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, small_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, h_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, v_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, big_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, huge_kernel)
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, medium_kernel)

    imagen = fill_holes(morph)

    rows,cols= imagen.shape
    M = np.float32([[1,0,-5],[0,1,-5]])
    imagen = cv.warpAffine(imagen,M,(cols,rows))

    return imagen

def morph_method2(im):
    imggray = im.copy()
    morph = imggray.copy()
    morph = cv.erode(morph,(2,2),iterations=1)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, huge_kernel)
    morph = fill_holes(morph)

    return morph

def morph_method3(im):
    imggray = im.copy()
    morph = imggray.copy()
    morph = cv.erode(morph,(3,3),iterations=1)

    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, big_kernel)
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, small_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, big_kernel)
    morph = fill_holes(morph)

    return morph

def morph_method4(im):
    imggray = im.copy()
    morph = imggray.copy()
    morph = cv.erode(morph,(2,2),iterations=1)

    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, huge_kernel)
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, small_kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, huge_kernel)
    morph = fill_holes(morph)

    return morph



def apply_morphology(im, morphology):
    switcher_morf = {
        'm1': morph_method1,
        'm2': morph_method2,
        'm3': morph_method3,
        'm4': morph_method4
    }
    
    # PIXEL MORPHOLOGY
    if morphology is not None:
        if not isinstance(morphology, list):
            morphology = list(morphology)
        for preproc in morphology:
            func = switcher_morf.get(preproc, lambda: "Invalid morphology")
            im = func(im)
    return im
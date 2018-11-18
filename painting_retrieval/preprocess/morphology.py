# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:09:49 2018

@author: hamdd
"""
import numpy as np
import cv2 as cv
from preprocess.utils import add_margin

def fill_holes(im):
    _, contours, _ = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(im, [cnt], 0, 255, -1)
    return im


def morph_method1(im):
    imggray = im.copy()

    morph = imggray.copy()


#    vk = lambda x : np.ones((x, 1), np.uint8)
#    hk = lambda x : np.ones((1, x), np.uint8)
    kk = lambda x : np.ones((x, x), np.uint8)

    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kk(15))
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kk(5))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kk(10))
    
    imagen = morph
    #imagen = fill_holes(morph)
    
    #rows, cols = imagen.shape
    #M = np.float32([[1, 0, -5], [0, 1, -5]])
    #imagen = cv.warpAffine(imagen, M, (cols, rows))
    return imagen

def morph_method2(im):
    imggray = im.copy()

    morph = imggray.copy()


    vk = lambda x : np.ones((x, 1), np.uint8)
    hk = lambda x : np.ones((1, x), np.uint8)
#    kk = lambda x : np.ones((x, x), np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, vk(23))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, hk(23))
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, hk(40))
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, vk(40))
    
    #### DO CCL ###
#    bb_list = bounding_box_utils(morph)
#    for bb in bb_list:
        
    imagen = morph
    #imagen = fill_holes(morph)
    
    #rows, cols = imagen.shape
    #M = np.float32([[1, 0, -5], [0, 1, -5]])
    #imagen = cv.warpAffine(imagen, M, (cols, rows))
    return imagen

def get_contours1(im, debug=False, add_str_debug=""):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    edges = cv.Canny(gray, 0, 40, apertureSize=3, L2gradient=True)
    morph_img = morph_method1(edges)
    fh_img = fill_holes(morph_img)
    edges2 = cv.Canny(fh_img, 60, 80, apertureSize=3, L2gradient=True)
    edges2 = cv.dilate(edges2,(3,3),iterations = 1)
    if(debug): 
        cv.imshow('Canny'+add_str_debug, edges)
        cv.imshow('fill_holes'+add_str_debug, fh_img)
        cv.imshow('Canny_2'+add_str_debug, edges2)
    
    return edges2, fh_img

def get_contours2(filled, debug=False, add_str_debug=""):
#    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#    gray = cv.GaussianBlur(edges, (11, 11), 0)
#    edges = cv.Canny(gray, 0, 40, apertureSize=3, L2gradient=True)
#    morph_img = morph_method2(edges)
#    fh_img = fill_holes(morph_img)
    madd = 1
    edges = add_margin(filled, madd).astype(np.uint8)
    edges2 = cv.Canny(edges, 60, 80, apertureSize=3, L2gradient=True)
    edges2 = cv.dilate(edges2,(3,3),iterations = 1)
    edges2 = edges2[madd:-(madd+1), madd:-(madd+1)]
    if(debug): 
#        cv.imshow('Canny'+add_str_debug, edges)
#        cv.imshow('fill_holes'+add_str_debug, fh_img)
        cv.imshow('Canny_2'+add_str_debug, edges2)
    
    return edges2
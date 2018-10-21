# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:55:09 2018

@author: hamdd
"""
import numpy as np

def boundingBoxFilter_method1(im, bb_list):
    image = im.copy()
    new_bb_list = []
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 700 or w*h > 20000 or f_ratio < 0.3 or form_factor < 0.333 or form_factor > 3):
            image[y:y+h, x:x+w] = np.zeros((h,w))
        else:
            new_bb_list.append((x,y,w,h))

    return image, new_bb_list

def boundingBoxFilter_method2(im, bb_list):
    image = im.copy()
    new_bb_list = []
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 700 or w*h > 20000 or f_ratio < 0.3 or form_factor < 0.5 or form_factor > 2.5):
            image[y:y+h, x:x+w] = np.zeros((h,w))
        else:
            new_bb_list.append((x,y,w,h))

    return image, new_bb_list

def boundingBoxFilter_method3(im, bb_list):
    image = im.copy()
    new_bb_list = []
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 1000 or w*h > 20000 or f_ratio < 0.35 or form_factor < 0.5 or form_factor > 2.5):
            image[y:y+h, x:x+w] = np.zeros((h,w))
        else:
            new_bb_list.append((x,y,w,h))

    return image, new_bb_list

def boundingBoxFilter_method4(im, bb_list):
    image = im.copy()
    new_bb_list = []
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 1000 or w*h > 25000 or f_ratio < 0.35 or form_factor < 0.5 or form_factor > 2.5):
            image[y:y+h, x:x+w] = np.zeros((h,w))
        else:
            new_bb_list.append((x,y,w,h))

    return image, new_bb_list



def filter_windows(bb_list, im, window_filter):
    switcher_window = {
        'm1': boundingBoxFilter_method1,
        'm2': boundingBoxFilter_method2,
        'm3': boundingBoxFilter_method3,
        'm4': boundingBoxFilter_method4

    }
        
    # PIXEL WINDOW FILTER
    if window_filter is not None and bb_list is not None:
        if not isinstance(window_filter, list):
            window_filter = list(window_filter)
        for preproc in window_filter:
            func = switcher_window.get(preproc, lambda: "Invalid window")
            im, bb_list = func(im, bb_list)
    
    return im, bb_list
    
    

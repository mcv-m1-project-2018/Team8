# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:57:43 2018

@author: hamdd
"""
import numpy as np 

def add_margin(mat, margin=1):
    padding = margin+2
    h, w = mat.shape
    zeroes = np.zeros((h+padding, w+padding))
    zeroes[margin:-(padding-margin), margin:-(padding-margin)] = mat
    return zeroes

def get_center_diff(imgrot, imgorig):
    rot_h, rot_w = imgrot.shape
    old_h, old_w = imgorig.shape[:2]
    rot_cp = (rot_w/2, rot_h/2)
    old_cp = (old_w/2, old_h/2)
    c_diff = (int(old_cp[0]-rot_cp[0]), int(old_cp[1]-rot_cp[1]))
    return c_diff
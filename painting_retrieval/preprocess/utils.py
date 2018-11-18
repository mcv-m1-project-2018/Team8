# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:57:43 2018

@author: hamdd
"""
from math import radians, sin, cos #, degrees
import numpy as np 
import cv2 as cv

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

def rotate_point(point, angle, center_point=(0, 0), convert_ints = True):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    x = new_point[0] + center_point[0]
    y = new_point[1] + center_point[1]
    if(convert_ints):
        x = int(x)
        y = int(y)
    new_point = (x, y)
    return new_point

def rotate_points(point_list, angle, center_point):
    rpoints = [rotate_point(p, angle, center_point) for p in point_list]
    return rpoints

def shrink(x, marg_low, marg_top):
    return min(max(x,marg_low),marg_top)
    
def correct_point(point, w, h):
    x = shrink(point[0], 0, w)
    y = shrink(point[1], 0, h)
    return (x,y)

def resize_keeping_ar(im, max_lateral=300):
    height, width = im.shape[:2]
    
    if(width > height):
        factor = width/float(max_lateral)
        desired_width = max_lateral
        desired_height = int(height/factor)
    else:
        factor = height/float(max_lateral)
        desired_height = max_lateral
        desired_width = int(width/factor)
    imres = cv.resize(im, (desired_width, desired_height))
    return imres, factor
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

import imutils
def rotate_and_crop(img, ang, pts):
    old_h,old_w = img.shape[:2]
#                cv2.imshow("original img", img)
#                img_orig = img.copy()
    img = imutils.rotate_bound(img, ang+180)
    rot_h,rot_w = img.shape[:2]
    old_c = (old_w/2, old_h/2)
    rot_c = (rot_w/2, rot_h/2)
    dif_c = (rot_c[0]-old_c[0],rot_c[1]-old_c[1])
    print(dif_c)
    
    ####### PRINT OVER UNROTATED
#                    img2 = img_orig.copy()
#                    radius = int(max(old_w/100, 5))
#                    thick = int(max(old_w/60-1, 5-1))
#                    color = (0,0,255)
#                    
#                    cv.circle(img2, pts[0], radius, color, thickness=thick)
#                    cv.circle(img2, pts[1], radius, color, thickness=thick)
#                    cv.circle(img2, pts[2], radius, color, thickness=thick)
#                    cv.circle(img2, pts[3], radius, color, thickness=thick)
#                    cv.imshow('Unrotated rectangle',resize_keeping_ar(img2)[0])
    ############################
    inc_point = lambda p: (int(p[0]+dif_c[0]), int(p[1]+dif_c[1]))
    rpts = rotate_points(pts, ang+180, old_c)
    rpts = [inc_point(p) for p in rpts]
    maxX = max([p[0] for p in rpts])
    minX = min([p[0] for p in rpts])
    maxY = max([p[1] for p in rpts])
    minY = min([p[1] for p in rpts])
#                    bX = int(minX+dif_c[0])
#                    tX = int(maxX+dif_c[0])
#                    bY = int(minY+dif_c[1])
#                    tY = int(maxY+dif_c[1])
    
    ####### PRINT OVER ROTATED
#                    img3 = img.copy()
#                    cv.circle(img3, rpts[0], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[1], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[2], radius, color, thickness=thick)
#                    cv.circle(img3, rpts[3], radius, color, thickness=thick)
#                    cv.imshow('rotated rectangle',resize_keeping_ar(img3)[0])
    ##########################
    img = img[minY:maxY, minX:maxX]
    return img
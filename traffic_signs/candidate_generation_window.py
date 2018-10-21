#!/usr/bin/python
import cv2 as cv
import numpy as np
# from numba import jit
import time
from math import floor

def is_intersect(x1,y1,w1,h1,x2,y2,w2,h2):
    is_intersecting = True
    if(x1 > x2+w2 or x1+w1 < x2):
        is_intersecting = False
    if(y1 > y2+h2 or y1+h1 < y2):
        is_intersecting = False
    return is_intersecting

def overlapped_windows(bb_list):
    # bb_overlaped = (bb,[related_bb_index])

    bb_overlapped = list()
    for x1,y1,w1,h1 in bb_list:
        related_bb_index = list()
        i=0
        for (x2,y2,w2,h2),_ in bb_overlapped:
            if is_intersect(x1,y1,w1,h1,x2,y2,w2,h2):
                related_bb_index.append(i)
            i +=1
        bb_overlapped.append(((x1,y1,w1,h1),related_bb_index))

    corespondence_dict = dict()
    bb_cell = list()
    i = 0
    for bb,index_list in bb_overlapped:
        if(len(index_list) > 0):
            bb_cell[corespondence_dict[index_list[0]]].append(bb)
            corespondence_dict[i] = corespondence_dict[index_list[0]]
        else:
            bb_cell.append([bb])
            corespondence_dict[i] = len(bb_cell)-1 #esto habria que hacerlo para todas las posiciones de la lista
        i += 1

    final_bbs = list()
    for group in bb_cell:
        x, y, x2, y2 = float('inf'),float('inf'),0,0
        for bb in group:
            x = min(bb[0],x)
            y = min(bb[1],y)
            x2 = max(bb[0]+bb[2], x2)
            y2 = max(bb[1]+bb[3], y2)
        final_bbs.append((x,y,x2-x,y2-y))

    im = np.zeros((500,500))
    for x,y,w,h in final_bbs:
       cv.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)

    cv.imshow('finish', im)
    cv.waitKey()

    return final_bbs

def overlapped_windows2(bb_list):
    bb_cell = list()

    while(len(bb_list)>0):
        x1,y1,w1,h1 = bb_list[0]
        del bb_list[0]
        hasModified = True
        while(hasModified):
            hasModified = False
            i = 0
            for x2,y2,w2,h2 in bb_list:
                if(is_intersect(x1,y1,w1,h1,x2,y2,w2,h2)):
                    w1 = max(x1+w1, x2+w2)
                    h1 = max(y1+h1, y2+h2)
                    x1 = min(x1,x2)
                    y1 = min(y1,y2)
                    w1 -= x1
                    h1 -= y1
                    del bb_list[i]
                    hasModified = True
                i +=1
        bb_cell.append((x1,y1,w1,h1))
    return bb_cell
            

def reduce_win_size(w, img):
    small_img = img[w[1]:(w[1]+w[3]),w[0]:(w[0]+w[2])]
    column = small_img.sum(0)
    row =    small_img.sum(1)
    
    new_x = 0
    new_w = 1
    new_y = 0
    new_h = 1

    while not column[ new_x]: new_x+=1
    while not column[-new_w]: new_w+=1
    while not    row[ new_y]: new_y+=1
    while not    row[-new_h]: new_h+=1

    new_win = (w[0] + new_x , w[1] + new_y, w[2] - new_w - new_x + 1, w[3] - new_h - new_y+1)
    
    return new_win

def reduce_winds_sizes(windows, img):
    for i, window in enumerate(windows):
        windows[i] = reduce_win_size(window, img)
    return windows

def boundingBox_ccl(im):
    # im = cv.cvtColor(im.copy(), cv.COLOR_RGB2GRAY)
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    bb_list = list()
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        bb_list.append((x,y,w,h))
    return bb_list

def boundingBox_sw(im):
    # window with anchor on top left point
    from main import CONSOLE_ARGUMENTS
    sw_size = CONSOLE_ARGUMENTS.window_size
    bb_list = list()
    n, m = im.shape
    sw_size = 45 #args Dani needed
    step = 8
    for x in range(0, m-sw_size, step):
        for y in range(0, n-sw_size, step):
            #print(x,x+sw_size,y,y+sw_size) #The output coordinates are given as x1,x2,y1,y2
            window_img = im[y:y+sw_size,x:x+sw_size]
            fRatio = np.count_nonzero(window_img)/(sw_size*sw_size)
            if(fRatio > 0.5):
                bb_list.append((x,y,sw_size,sw_size))
    newbb = overlapped_windows2(bb_list)
    return newbb

def boundingBox_sw_integrate(im):
    # window with anchor on top left point
    bb_list = list()
    n, m = im.shape
    mask = im[:,:]/255
    from main import CONSOLE_ARGUMENTS
    sw_size = CONSOLE_ARGUMENTS.window_size
    step = 8
    ii = np.zeros((n,m))
    s = np.zeros((n,m))
    for y in range(0, n):
        for x in range(1, m):
            s[y,x] = mask[y,x] + s[y,x-1]
            if(y == 0):
                ii[y,x] = s[y,x]
            else:
                ii[y,x] = s[y,x] + ii[y-1,x]

    box_size = 1/(sw_size*sw_size)
    for y in range(0,n-sw_size, step):
        for x in range(0, m-sw_size, step):
            window_img = ii[y+sw_size,x+sw_size] - ii[y,x+sw_size] - ii[y+sw_size,x] + ii[y,x]
            fRatio = window_img*box_size
            if(fRatio > 0.5):
                bb_list.append((x,y,sw_size,sw_size))
    newbb = overlapped_windows2(bb_list)
    return newbb

def boundingBox_sw_conv(im):
    # window with anchor on top left point
    bb_list = list()
    n, m = im.shape
    from main import CONSOLE_ARGUMENTS
    sw_size = CONSOLE_ARGUMENTS.window_size
    step = 1

    a = cv.filter2D(im, cv.CV_32F, np.ones((1,sw_size)),borderType=cv.BORDER_CONSTANT)
    a = cv.filter2D(a, cv.CV_32F, np.ones((sw_size,1)),borderType=cv.BORDER_CONSTANT)/(sw_size*sw_size*255)

    for x in range(0, m-sw_size, step):
        for y in range(0, n-sw_size, step):
            fRatio = a[(y+floor(sw_size/2)),(x+floor(sw_size/2))]
            if(fRatio > 0.5):
                bb_list.append([x,y,sw_size,sw_size])
    newbb = overlapped_windows2(bb_list)

    return newbb


# Create your own candidate_generation_window_xxx functions for other methods
# Add them to the switcher dictionary in the switch_method() function
# These functions should take an image, a pixel_candidates mask (and perhaps other parameters) as input and output the window_candidates list.

def generate_windows(im, method, reduce_bbs=True):
    return switch_method(im, method, reduce_bbs=reduce_bbs)

def switch_method(im, method, reduce_bbs=True):
    bb_list = None
    switcher_bb = {
        'ccl': boundingBox_ccl,
        'sw': boundingBox_sw,
        'swi': boundingBox_sw_integrate,
        'swconv': boundingBox_sw_conv
    }
    # Get the function from switcher dictionary
    if method is not None:
        if not isinstance(method, list):
            method = list(method)
        for preproc in method:
            func = switcher_bb.get(preproc, lambda: "Invalid bounding box")
            bb_list = func(im)
            if(reduce_bbs): bb_list = reduce_winds_sizes(bb_list, im)
    return bb_list


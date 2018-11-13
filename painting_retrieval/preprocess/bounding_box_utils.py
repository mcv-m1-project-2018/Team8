
import cv2 as cv
from tqdm import tqdm
import numpy as np

def is_intersect(x1,y1,w1,h1,x2,y2,w2,h2):
    is_intersecting = True
    if(x1 > x2+w2 or x1+w1 < x2):
        is_intersecting = False
    if(y1 > y2+h2 or y1+h1 < y2):
        is_intersecting = False
    return is_intersecting

def boundingBox_ccl(im):
    # im = cv.cvtColor(im.copy(), cv.COLOR_RGB2GRAY)
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    bb_list = list()
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        bb_list.append((x,y,w,h))
    return bb_list

def overlapped_windows(bb_list):
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


def imshow_bb(im, bb_list):
    image = im.copy()
    if(len(bb_list) > 0):
        for i in range(len(bb_list)):
            if(bb_list[i] != None):
                x,y,w,h = bb_list[i]
                cv.rectangle(image,(x,y),(x+w,y+h),(200,0,0),2)

    image = cv.resize(image,None, fx=0.5, fy=0.5)
    cv.imshow("bb",image)
    cv.waitKey()
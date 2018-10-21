# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv

def template_matching_with_metrics(mask, bb_list):
    """
    bb_classified = list of (x,y,w,h,nameType)
    """
    bb_classified = list()
    for x,y,w,h in bb_list:
        nameType = "none"
        window = mask[y:y+h,x:x+w]
        n_white = np.count_nonzero(window)
        fratio = n_white/float(w*h)
        if(fratio<0.60 and fratio>0.40):
            n_white_top = np.count_nonzero(window[:round(h/2),:])
            n_white_bottom = np.count_nonzero(window[round(h/2)+1:,:])
            if (n_white_top > n_white_bottom):
                #If the top is bigger than the bottom is a yield-like signal 
                nameType = "B"
            else:
                nameType = "A"
        elif(fratio<0.85 and fratio>0.7):
            nameType = "CDE"
        elif(fratio > 0.85):
            nameType = "F"
        bb_classified.append((x,y,w,h,nameType))
    return bb_classified

def template_matching_with_correlation(mask, bb_list, non_affinity_removal=False):
    """
    bb_classified = list of (x,y,w,h,type)
    """
    from main import CONSOLE_ARGUMENTS
    thr1 = CONSOLE_ARGUMENTS.non_affinity_removal1
    thr2 = CONSOLE_ARGUMENTS.non_affinity_removal2
    index2Class = {0:"A",1:"B",2:"CDE",3:"F"}
    bb_classified = list()
    for x,y,w,h in bb_list:
        window = mask[y:y+h,x:x+w]
        F_template = np.ones((h,w),dtype='uint8')*255
        CDE_template = cv.circle(np.zeros((h,w),dtype='uint8'),(round(w/2),round(h/2)), round(min(h,w)/2),(255,255,255),thickness=-1)
        # CDE_template = cv.ellipse(np.zeros((h,w)), (x,y,w,h), (255,255,255))
        A_template = cv.fillPoly(np.zeros((h,w),dtype='uint8'),[np.array([[round(w/2),0],[w,h],[0,h]])],(255,255,255))
        B_template = cv.fillPoly(np.zeros((h,w),dtype='uint8'),[np.array([[0,0],[w,0],[round(w/2),h]])],(255,255,255))
        # cv.imshow("asd",CDE_template)
        # cv.waitKey()
        out_F = cv.matchTemplate(window,F_template,cv.TM_CCORR_NORMED)
        out_CDE = cv.matchTemplate(window,CDE_template,cv.TM_CCORR_NORMED)
        out_A = cv.matchTemplate(window,A_template,cv.TM_CCORR_NORMED)
        out_B = cv.matchTemplate(window,B_template,cv.TM_CCORR_NORMED)
#        print("Result:", out_F, out_CDE,out_A,out_B)
        maximum = np.max([out_A,out_B,out_CDE,out_F])
        if(non_affinity_removal and (h < 60 or w < 60) and maximum < thr1):
            mask[y:y+h,x:x+w] = np.zeros((h,w))
        elif(non_affinity_removal and maximum < thr2 and len(bb_list) > 1):
            mask[y:y+h,x:x+w] = np.zeros((h,w))
#            print("deleted:", (x,y,w,h))
        else:
            nameType = index2Class[np.argmax([out_A,out_B,out_CDE,out_F])]
            bb_classified.append((x,y,w,h,nameType))

    return bb_classified

def template_matching(im, bb_list, non_affinity_removal=False):
    modImage = im[:,:]
    bb_class_list = None
    if(bb_list is not None):
        bb_class_list = template_matching_with_correlation(modImage,bb_list, \
                                                           non_affinity_removal=non_affinity_removal)
    return bb_class_list
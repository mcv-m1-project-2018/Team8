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

def template_matching_with_correlation(mask, bb_list):
    """
    bb_classified = list of (x,y,w,h,type)
    """
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
        nameType = index2Class[np.argmax([out_A,out_B,out_CDE,out_F])]
        bb_classified.append((x,y,w,h,nameType))

    return bb_classified

def template_matching(im, bb_list):
    if(bb_list is not None):
        bb_class_list = template_matching_with_correlation(im,bb_list)
        for x,y,w,h,name in bb_class_list:
            cv.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
            cv.putText(im,name,(x,y), cv.QT_FONT_NORMAL, 1,(150,150,150),2,cv.LINE_AA)
            # cv.putText(pixel_candidates[y:y+h,x:x+w], name,cv.QT_FONT_NORMAL, 2, (0,255,0))
    return im
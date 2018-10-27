# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:23:01 2018

@author: hamdd
"""

import cv2 as cv

def equalization(hist):
#    cv.NormalizeHist(hist, factor)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    return cdf_normalized

def correlation(hist1, hist2):
    pass
def preprocess(hist1, hist2, mode):
    switcher = {"equalize": equalization,
                "equalization": equalization,
                "correlate":    correlation,
                "correlation":  correlation
                }
    
    
    if mode is not None:
        if not isinstance(mode, list):
            mode = list(mode)
        for preproc in mode:
            if(preproc in ["equalize","equalization"]):
                hist1 = equalization(hist1)
                hist2 = equalization(hist2)
            elif(preproc in ["correlate","correlation"]):
                hist1, hist2 = correlation(hist1, hist2)
                
    return hist1, hist2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:23:01 2018

@author: hamdd
"""

import cv2 as cv
from tqdm import tqdm 
import numpy as np

def equalization(hist):
#    cv.NormalizeHist(hist, factor)
    for i, ch in enumerate(hist):
        minel = sum(ch) // len(ch)
        minel = minel.astype(np.int)
        numex = sum(ch) - minel * len(ch)
        numex = numex.astype(np.int)
        hist[i] = [minel+1] * numex + [minel] * (len(ch) - numex)
#        cdf = ch.cumsum()
#        cdf_normalized = cdf * ch.max()/ cdf.max()
    return hist

def normalization(hist):
    for i, ch in enumerate(hist):
        hist[i] = [float(i)/sum(ch) for i in ch]
    return hist

def preprocess(hist, mode):
    switcher = {equalization:["equalize","equalization","eq", "equal"],
                normalization:["normalize","normalization","norm", "normal"]
                }
    
    if mode is not None:
        if not (isinstance(mode, list)):
            mode = [mode]
        for preproc in mode:
            for func, names in switcher.items():
                if(preproc in names):
                    hist = func(hist)
                
    return hist

def preprocessAllHistograms(histList, mode):
    for i, hist in enumerate(histList):
        histList[i] = preprocess(hist, mode)
    return histList
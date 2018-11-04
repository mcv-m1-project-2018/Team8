import numpy as np
#import cv2 as cv
#import ml_metrics as metrics
import cv2
# CV2 comparison methods guide:
# https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#gga994f53817d621e2e4228fc646342d386a035d9882661edb22a95895516e441428

def euclidean(t, q): return pow(t - q, 2)
def        L1(t, q): return abs(t - q)
def      x_sq(t, q): return pow(t - q, 2) / ((t + q) if t+q else 1)
def  hist_int(t, q): return min(t, q)
def  kernhell(t, q): return t*q
def bhattacharyya(t,q): return cv2.compareHist(t, q, cv2.HISTCMP_BHATTACHARYYA)
def  x_sq_alt(t,q): return cv2.compareHist(t, q, cv2.HISTCMP_CHISQR_ALT)
def    kl_div(t,q): return cv2.compareHist(t, q, cv2.HISTCMP_KL_DIV)
def correlate(t,q): return cv2.compareHist(t, q, cv2.HISTCMP_CORREL)

def evaluate(t_bins, q_bins, eval_type):
    #INITIALIZATION
    switcher = {"euclidean": euclidean,
                "L1": L1,
                "x_sq": x_sq,
                "hist_intersect": hist_int,
                "kernhell": kernhell,
                "bhattacharyya":bhattacharyya,
                "x_sq_alt":x_sq_alt,
                "kl_div":kl_div,
                "correlate": correlate
                }
                
    #ERROR CHECK
    if(eval_type not in switcher.keys()):
        raise(ValueError("Evaluation type does not correspond with any known \
                         implemented function: \""+eval_type+"\" which should\
                         be one of the following:", switcher.keys()))
    
    if(len(t_bins) != len(q_bins)):
        raise(ValueError("Len of both bins does not match: ", len(t_bins), 
                         " vs ", len(q_bins), "(T and B)"))
    
    #SUMATORY
    dist = 0
    if(eval_type not in ["bhattacharyya","x_sq_alt","kl_div","correlate"]):
        for i in range(len(t_bins)):
            dist += switcher[eval_type](t_bins[i], q_bins[i])
    else:
        dist = switcher[eval_type](np.array(t_bins), np.array(q_bins))
        
    #SQRT
    if(eval_type in ["euclidean", "kernhell"]):
        dist = np.sqrt(dist)
    return dist
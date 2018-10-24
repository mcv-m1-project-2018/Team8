import numpy as np
#import cv2 as cv
#import ml_metrics as metrics

def euclidean(t, q): return pow(t - q, 2)
def        L1(t, q): return abs(t - q)
def      x_sq(t, q): return pow(t - q, 2) / ((t + q) if t+q else 1)
def  hist_int(t, q): return min(t, q)
def  kernhell(t, q): return t*q
    
def evaluate(t_bins, q_bins, eval_type):
    #INITIALIZATION
    switcher = {"euclidean": euclidean,
                "L1": L1,
                "x_sq": x_sq,
                "hist_intersect": hist_int,
                "kernhell": kernhell
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
    for i in range(len(t_bins)):
        dist += switcher[eval_type](t_bins[i], q_bins[i])
    
    #SQRT
    if(eval_type in ["euclidean", "kernhell"]):
        dist = np.sqrt(dist)
    return dist
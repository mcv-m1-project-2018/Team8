# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 18:51:59 2018

@author: hamdd
"""
import numpy as np
import cv2 as cv
lib = cv.xfeatures2d

class azemar_create:
    def __init__(self):
        pass
    def detectAndCompute(self,img, dafok):
        print("THIS IS A JOKE JAJA")
        return [(0,0)], np.array([0,2,3,1,4,52,2])
    
detector_s = {
            "SIFT":lib.SIFT_create,
            "SURF":lib.SURF_create,
            "AKAZE":cv.AKAZE_create,
            "BRISK":cv.BRISK_create,
            "KAZE":cv.KAZE_create,
            "MSER":cv.MSER_create,
            "ORB":cv.ORB_create,
            "StarDetector":lib.StarDetector_create
            }

compute_s = {
            "BoostDesc" : lib.BoostDesc_create,
            "BriefDescriptorExtractor":lib.BriefDescriptorExtractor_create,
            "DAISY":lib.DAISY_create,
            "FREAK":lib.FREAK_create,
            "HarrisLaplaceFeatureDetector":lib.HarrisLaplaceFeatureDetector_create,
            "LATCH":lib.LATCH_create,
            "LUCID":lib.LUCID_create,
            "PCTSignaturesSQFD":lib.PCTSignaturesSQFD_create,
            "PCTSignatures":lib.PCTSignatures_create,
            "StarDetector":lib.StarDetector_create,
            "VGG": lib.VGG_create,
            "AZEMAR": azemar_create,
            "Agast":cv.AgastFeatureDetector_create,
            "FAST":cv.FastFeatureDetector_create,
            "GFTT":cv.GFTTDetector_create,
        }

compute_s.update(detector_s)

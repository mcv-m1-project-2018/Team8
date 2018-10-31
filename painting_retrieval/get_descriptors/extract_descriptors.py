# -*- coding: utf-8 -*-
import cv2 as cv

def extract_features(img, descriptor, colorspace="gray"):
    if(colorspace=="gray"):
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    lib = cv.xfeatures2d
    switcher = {"BoostDesc" : lib.BoostDesc_create,
                "BriefDescriptorExtractor":lib.BriefDescriptorExtractor_create,
                "DAISY":lib.DAISY_create,
                "FREAK":lib.FREAK_create,
                "HarrisLaplaceFeatureDetector":lib.HarrisLaplaceFeatureDetector_create,
                "LATCH":lib.LATCH_create,
                "LUCID":lib.LUCID_create,
                "PCTSignaturesSQFD":lib.PCTSignaturesSQFD_create,
                "PCTSignatures":lib.PCTSignatures_create,
                "StarDetector":lib.StarDetector_create,
                "VGG":lib.VGG_create,
                
                "SIFT":lib.SIFT_create,
                "SURF":lib.SURF_create,
                "Agast":cv.AgastFeatureDetector_create,
                "AKAZE":cv.AKAZE_create,
                "BRISK":cv.BRISK_create,
                "FAST":cv.FastFeatureDetector_create,
                "GFTT":cv.GFTTDetector_create,
                "KAZE":cv.KAZE_create,
                "MSER":cv.MSER_create,
                "ORB":cv.ORB_create
            }
    descgen = switcher[descriptor]
    kp,desc = descgen.detectAndCompute(img,None)
    return kp, desc
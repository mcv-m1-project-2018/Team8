# -*- coding: utf-8 -*-
import cv2 as cv

class azemar_create:
    def __init__(self):
        pass
    def detectAndCompute(self,img, dafok):
        print("THIS IS A JOKE JAJA")
        return [(0,0)], np.array([0,2,3,1,4,52,2])
    
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
                "SIFT":lib.SIFT_create,
                "SURF":lib.SURF_create,
                "StarDetector":lib.StarDetector_create,
                "VGG": lib.VGG_create,
                "AZEMAR": azemar_create
            }
    descgen = switcher[descriptor]()
    kp,desc = descgen.detectAndCompute(img,None)
    return kp, desc

def extract_all_features(names,path, descriptor, colorspace="gray"):
    desc_all = []
    kp_all = []
    for name in names:
        img = cv.imread(path+name)
        kp, desc = extract_features(img, descriptor, colorspace=colorspace)
        desc_all.append(desc)
        kp_all.append(kp)
    return kp_all, desc_all
import cv2 as cv
from tqdm import tqdm
import numpy as np

def detectBoxes(file_names, image_path):
    for name in tqdm(file_names):
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
    
        thr_tophat = 100
        kernel = np.ones((20,20))
        res = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
        res = (res[:,:,0]>thr_tophat)*(res[:,:,1]>thr_tophat)*(res[:,:,2]>thr_tophat)
        res = np.dstack((res,res,res))*image
        res2 = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)

        thr_val = 180
        thr = (res[:,:,0]>thr_val)*(res[:,:,1]>thr_val)*(res[:,:,2]>thr_val)
        # thr = thr + (res[:,:,0]<55)*(res[:,:,1]<55)*(res[:,:,2]<55)
        thr = np.dstack((thr,thr,thr)) * image

        res = cv.resize(res,None, fx=0.5, fy=0.5)
        res2 = cv.resize(res2,None, fx=0.5, fy=0.5)
        thr = cv.resize(thr,None, fx=0.5, fy=0.5)
        cv.imshow('res',res)
        cv.imshow('res2',res2)
        cv.imshow('thr',thr)
        cv.waitKey()
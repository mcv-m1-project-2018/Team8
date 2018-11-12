import cv2 as cv
from tqdm import tqdm
import numpy as np

def detectBoxes(file_names, image_path):
    for name in tqdm(file_names):
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
    
        kernel = np.ones((20,20))

        thr_tophat = 100
        res = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
        res = (res[:,:,0]>thr_tophat)*(res[:,:,1]>thr_tophat)*(res[:,:,2]>thr_tophat)
        res = np.dstack((res,res,res))*image

        thr_blackhat = 150
        res2 = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
        black_thr = (res2[:,:,0]>thr_blackhat)*(res2[:,:,1]>thr_blackhat)*(res2[:,:,2]>thr_blackhat)
        res2 = np.dstack((black_thr,black_thr,black_thr)) * res2
        

        thr_val = 180
        thr = (res[:,:,0]>thr_val)*(res[:,:,1]>thr_val)*(res[:,:,2]>thr_val)
        thr = np.dstack((thr,thr,thr)) * image

        thr_val = 130
        thr2 = (res2[:,:,0]>thr_val)*(res2[:,:,1]>thr_val)*(res2[:,:,2]>thr_val)
        thr2 = np.dstack((thr2,thr2,thr2)) * image

        res = cv.resize(res,None, fx=0.5, fy=0.5)
        res2 = cv.resize(res2,None, fx=0.5, fy=0.5)
        thr = cv.resize(thr,None, fx=0.5, fy=0.5)
        thr2 = cv.resize(thr2,None, fx=0.5, fy=0.5)
        cv.imshow('top',res)
        cv.imshow('black',res2)
        cv.imshow('topthr',thr)
        cv.imshow('blackthr',thr2)
        cv.waitKey()
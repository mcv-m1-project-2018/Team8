import cv2 as cv
from tqdm import tqdm
import numpy as np

from sklearn.cluster import MeanShift


def detect_text_hats(file_names, image_path):
    n_images = len(file_names)
#     for name in tqdm(file_names):
    i = 0
    while i < n_images:
        name = file_names[i]
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
        thr = np.dstack((thr,thr,thr)).astype('uint8') * 255

        thr_val = 130
        thr2 = (res2[:,:,0]>thr_val)*(res2[:,:,1]>thr_val)*(res2[:,:,2]>thr_val)
        thr2 = np.dstack((thr2,thr2,thr2)).astype('uint8') *255 

        res = cv.resize(res,None, fx=0.5, fy=0.5)
        res2 = cv.resize(res2,None, fx=0.5, fy=0.5)
        thr = cv.resize(thr,None, fx=0.5, fy=0.5)
        thr2 = cv.resize(thr2,None, fx=0.5, fy=0.5)
        cv.imshow('top',res)
        cv.imshow('black',res2)
        cv.imshow('topthr',thr)
        cv.imshow('blackthr',thr2)
        k = cv.waitKey()
        
        if k==27 or k==-1:    # Esc key or close to stop
            break
        elif k==97 and i>0:    # A to go back
            i-=1
        else:                   # Aby key to go forward
            i+=1



def detect_text_meanShift(file_names, image_path):
    for name in tqdm(file_names):
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        imageArray = np.reshape(image,(-1,1))
        clustering = MeanShift(bandwidth=3).fit(imageArray)

        print(clustering.labels_)

        imageLabels = clustering.predict(imageArray)

        # thr2 = cv.resize(thr2,None, fx=0.5, fy=0.5)
        cv.imshow('blackthr',image)
        cv.waitKey()
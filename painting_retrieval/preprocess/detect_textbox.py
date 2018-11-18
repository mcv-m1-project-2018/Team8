import cv2 as cv
from tqdm import tqdm
import numpy as np
import pickle as pckl

from preprocess.bounding_box_utils import boundingBox_ccl,overlapped_windows,imshow_bb
from sklearn.cluster import MeanShift

import matplotlib.pyplot as plt
from preprocess.crop_and_rotate import resize_keeping_ar

def saveTextBoxArray(save_path, bb_list):
    pckl_file = open(save_path,"wb")
    pckl.dump(bb_list,pckl_file)
    pckl_file.close()

def generateMaskFrombb(bb, size):
    mask = np.zeros(size)
    x,y,w,h = bb
    mask = cv.rectangle(mask, (x,y), (x+w,y+h), (255,255,255), -1)
    mask = 255-mask
    mask = mask[:,:,0]
    return mask


def filter_bb(bb_list, im):
    new_bb = []
    for x,y,w,h in bb_list:
        # region = im[y:y+h,x:x+w]
        # totalPixels = np.count_nonzero(region)
        if( w/h >2.5 and h > 13 and w > 120):
            new_bb.append((x,y,w,h))

    return new_bb

def selectRealBoundingBox(bb_list, im):
    # (x,y,w,h)
    # really sofisticated algorithm
    if(len(bb_list) > 0):
        maxpixels = 0
        maxbb = ()
        for x,y,w,h in bb_list:
            region = im[y:y+h,x:x+w]
            totalPixels = np.count_nonzero(region)
            if(maxpixels < totalPixels):
                maxpixels = totalPixels
                maxbb = (x,y,w,h)

        return maxbb
    else:
        return (0,0,0,0)
    
def obtain_bb(image, debug=True):
    image = cv.GaussianBlur(image, (3,3),5)
    
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

    thr_val = 130
    thr2 = (res2[:,:,0]>thr_val)*(res2[:,:,1]>thr_val)*(res2[:,:,2]>thr_val)

    out1 = np.minimum(thr + thr2, 255)
    out1 = np.dstack((out1,out1,out1)).astype('uint8') * 255

    thr2 = np.dstack((thr2,thr2,thr2)).astype('uint8') *255 
    thr = np.dstack((thr,thr,thr)).astype('uint8') * 255

    
    out1 = cv.morphologyEx(out1, cv.MORPH_CLOSE, np.ones((5,5)))
    out1 = cv.morphologyEx(out1, cv.MORPH_OPEN, np.ones((3,3)))

    out1 = cv.erode(out1,np.ones((2,2)))
    # out1 = cv.dilate(out1,np.ones((1,100)))
    out1 = cv.morphologyEx(out1, cv.MORPH_CLOSE, np.ones((1,120)))

    bb_list = boundingBox_ccl(cv.cvtColor(out1,cv.COLOR_BGR2GRAY))

    bb_list = overlapped_windows(bb_list)
    bb_list = filter_bb(bb_list,out1)

        # imshow_bb(out1,[selectRealBoundingBox(bb_list,out1)])
    endBB = selectRealBoundingBox(bb_list,out1)
    
    if(debug):
        res = cv.resize(res,None, fx=0.5, fy=0.5)
        res2 = cv.resize(res2,None, fx=0.5, fy=0.5)
        thr = cv.resize(thr,None, fx=0.5, fy=0.5)
        thr2 = cv.resize(thr2,None, fx=0.5, fy=0.5)
        out1 = cv.resize(out1,None, fx=0.5, fy=0.5)
        cv.imshow('top',res)
        cv.imshow('black',res2)
        cv.imshow('topthr',thr)
        cv.imshow('blackthr',thr2)
        cv.imshow('top+black',out1)
    return endBB

def neutre(im, sz=10):  
    """
    Funcio que neutralitza el color de la imatge. Util quan es esta tacat o hi
    ha variancies.
    * Inputs:
    - im = skimage.io image
    - params = diccionari de parametres
    *Outputs:
    - im = imatge amb color neutralitzat
    """
    
    x, y = im.shape[:2]
    kernel = np.ones((sz,sz),np.uint8)
    resd = cv.erode(cv.dilate(np.int16(im),kernel,1),kernel,1)
    resd = np.array(resd, dtype=np.float)
    im = np.divide(im,resd)
    return im

def detect_text_hats(file_names, image_path, debug_text_bb_thresholds=False):
    n_images = len(file_names)
    bb_all_list = []
    bb_all_mask = []
#    for i in tqdm(range(len(file_names))):
    i = 0

    while i < n_images:
        print(i,file_names[i])
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
        image, factor = resize_keeping_ar(image)

        endbb = obtain_bb(image, debug_text_bb_thresholds)
#        x, y, w, h = endbb
#        m = 20
#        kk = lambda x: np.ones((x, x))
#        im_eroded = cv.erode(image,kk(10),iterations = 1)
#        column = im_eroded.sum(0).sum(1)
#        row = im_eroded.sum(1).sum(1)
#        plt.figure()
#        plt.plot(range(len(row)),row)
#        gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kk(10))
#        cv.imshow("gradient", gradient)
#        im_neutre = neutre(image_orig, 10)
#        cv.imshow("morph", image)
#        cv.imshow("neutre", im_neutre.astype(np.uint8))
#        im_bb = image[y:y+w,x:x+w]
#        im_bb_big = image[y-m:y+h+m,x-m:x+w+m, :]
#        maskbb = np.zeros_like(im_bb_big)
#        maskbb[m:-m, m:-m] = np.ones((h, w, 3))
#        masked_im = np.ma.array(im_bb_big, mask=maskbb.astype(np.bool))
#        cv.imshow("maskara", masked_im)
#
#        mean = []
#        mean.append(masked_im[:,:,0].mean())
#        mean.append(masked_im[:,:,1].mean())
#        mean.append(masked_im[:,:,2].mean())
#        
#        std = []
#        std.append(masked_im[:,:,0].std()*0.5)
#        std.append(masked_im[:,:,1].std()*0.5)
#        std.append(masked_im[:,:,2].std()*0.5)
#
#        
        bb_all_list.append(endbb)
        rare_mask = generateMaskFrombb(endbb,image.shape)
        bb_all_mask.append(rare_mask)

        if(debug_text_bb_thresholds):

            k = cv.waitKey()
    
            if k==27 or k==-1:    # Esc key or close to stop
                break
            elif k==97 and i>0:    # A to go back
                i-=1
            else:                   # Aby key to go forward
                i+=1
        else:
            i+=1
    return bb_all_list, bb_all_mask



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
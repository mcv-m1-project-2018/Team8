import pywt 
import numpy as np
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt

import multiprocessing as mp
from multiprocessing.pool import ApplyResult

def getGSum(img, morph_type, k):
    kernel = np.ones((k,k), np.uint8)
    morph = cv.morphologyEx(img, morph_type, kernel)
    return np.sum(morph, dtype=np.int64)

def visualizeHistogram(histogram,bin_num):
    plt.figure()
    plt.title("histogram")
    plt.ioff()
    plt.plot(histogram, color='r')
    plt.xlim([0, bin_num*2])
    plt.show()

def calculateGranulometry(image, bin_num):
    img = image[:,:,:]
    if(img.shape[2] >2):
        img = np.asarray(cv.cvtColor(img, cv.COLOR_BGR2GRAY),dtype='float')/255.0

#    valueList = np.zeros(bin_num*2)
#    lastImageValue = np.sum(img)
#    valueList[bin_num] = 0

    pool = mp.Pool(processes=mp.cpu_count())
    close_results = [pool.apply_async((getGSum),(image, cv.MORPH_CLOSE, k)) for k in range(1,bin_num+1)]
    open_results  = [pool.apply_async((getGSum),(image, cv.MORPH_OPEN , k)) for k in range(1,bin_num  )]
    
    pool.close()
    # waiting for all results
    map(ApplyResult.wait, close_results)
    map(ApplyResult.wait, open_results)
#    RESULT_LIST=[r.get() for r in async_results]
    
    close_results = [r.get() for r in close_results][::-1]
    open_results =  [r.get() for r in open_results ]

    before = close_results[0]
    for x in range(len(close_results)):
        diff = close_results[-x]-before
        before = close_results[-x]
        close_results[-x] = diff
    before =  open_results[0]
    for x in range(len(open_results )):
        diff = before-open_results[x]
        before = open_results[x]
        open_results[x] = diff
    
    open_results[0] = 0
    close_results[-1]=0
    vL = close_results+open_results
        
#    vL = tmp + [tmp=x-tmp for x in vL[1:]]
#    valueList
#    valueList.extend([r.get() for r in close_results])
#    valueList.extend([r.get() for r in open_results ])
#    sys.stdout.write(str(RESULT_FINAL))
    
#    for i in range(1,bin_num+1):
#        kernel = np.ones((i,i), np.uint8)
#        currentImageValue = np.sum(cv.morphologyEx(img, cv.MORPH_CLOSE, kernel))
#        valueList[bin_num-i] = currentImageValue - lastImageValue
#        lastImageValue = currentImageValue

#    lastImageValue = np.sum(img)
#    for i in range(1,bin_num):
#        kernel = np.ones((i,i), np.uint8)
#        currentImage = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#        currentImageValue = np.sum(currentImage)
#        valueList[i+bin_num] =  lastImageValue - currentImageValue
#        lastImageValue = currentImageValue
    
    return vL


def processGranulometry(file_names, imPath, bin_num, visualize):
    """
    Performs a granulometry for every image
    """ 
    histAll = list()
    for name in tqdm(file_names):
        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        imageHist = calculateGranulometry(image, bin_num)
        histAll.append([imageHist])
    
    if(visualize):
        visualizeHistogram(imageHist,bin_num)
    return histAll
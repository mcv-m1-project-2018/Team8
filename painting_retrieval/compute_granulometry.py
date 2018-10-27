import pywt 
import numpy as np
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt

import multiprocessing as mp

def visualizeHistogram(histogram,bin_num):
    plt.figure()
    plt.title("histogram")
    plt.ioff()
    plt.plot(histogram, color='r')
    plt.xlim([0, bin_num*2])
    plt.show()

def calculateGranulometry(image, bin_num):
    img = image[:,:,:]
    if(img.shape[2] >1):
        img = np.asarray(cv.cvtColor(img, cv.COLOR_BGR2GRAY),dtype='float')/255.0

    valueList = np.zeros(bin_num*2)
    lastImageValue = np.sum(img)
    valueList[bin_num] = 0

    for i in range(1,bin_num+1):
        kernel = np.ones((i,i), np.uint8)
        currentImageValue = np.sum(cv.morphologyEx(img, cv.MORPH_CLOSE, kernel))
        valueList[bin_num-i] = currentImageValue - lastImageValue
        lastImageValue = currentImageValue

    lastImageValue = np.sum(img)
    for i in range(1,bin_num):
        kernel = np.ones((i,i), np.uint8)
        currentImage = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        currentImageValue = np.sum(currentImage)
        valueList[i+bin_num] =  lastImageValue - currentImageValue
        lastImageValue = currentImageValue
    return valueList


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